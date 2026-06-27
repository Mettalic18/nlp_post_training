#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import torch
import json
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


def generate_responses(model, tokenizer, user_message, system_message=None, max_new_tokens=100):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )

    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def test_model_with_questions(model, tokenizer, questions, system_message = None):
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")


def load_model_and_tokenizer(model_name, use_gpu=False):
    config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)

    tokenizer.chat_template = """{% for message in messages %}
        {% if message['role'] == 'system' %}<|im_start|>system
        {{ message['content'] }}<|im_end|>
        {% elif message['role'] == 'user' %}<|im_start|>user
        {{ message['content'] }}<|im_end|>
        {% elif message['role'] == 'assistant' %}<|im_start|>assistant
        {{ message['content'] }}<|im_end|>
        {% endif %}
        {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        {% endif %}"""

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=torch.float32,
    )

    if use_gpu:
        model.to("cuda")

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.generation_config.do_sample = False
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)

    return model, tokenizer




def fchat_example(tokenizer):
    def format_chat_example(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize = False,
            add_generation_prompt = False,
            enable_thinking = False,
        )
    return format_chat_example


def save_finetuned_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))


def train(args):
    """Fine-tune a Qwen3 large language model."""
    print("=" * 60)
    print("FINE-TUNING")
    print("=" * 60)

    USE_GPU = torch.cuda.is_available()


    print(f"Using GPU: {USE_GPU}")
    print(f"Base model path: {args.model_load}")
    print(f"SFT model path: {args.model_save}")

    model, tokenizer = load_model_and_tokenizer(args.model_load, USE_GPU)

    # load the dataset from the JSON file
    train_dataset = load_dataset("json", data_files=args.train_data)["train"]

    # Show one raw dataset example and the corresponding formatted training text.
    j = 0
    ex_j = train_dataset[j]
    print("Raw dataset example:")
    print(ex_j)

    fex_j = fchat_example(tokenizer)(ex_j)
    print("\nFormatted example after applying fchat_example(tokenizer):")
    print(fex_j)
    

    sft_config = SFTConfig(
        output_dir = args.model_save,
        learning_rate = 8e-5,
        num_train_epochs = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        gradient_checkpointing = False,
        logging_steps = 2, 
    )
    # Here, 1 sample is loaded at a time on GPU.
    # Gradients are accumulated for 8 forward/backward passes before one optimizer update.
    # So the effective batch size per optimizer step is: per_device_train_batch_size * gradient_accumulation_steps = 1 * 8 = 8 samples.
    #
    # The training split has 2961 samples, so one epoch takes about:
    # 2961 / 8 = 370.125 optimizer steps, which becomes 371 steps in practice.
    #
    # logging_steps = 2 means training metrics are printed every 2 optimizer steps.
    # Since each optimizer step corresponds to 8 samples, logs appear after about: 2 * 8 = 16 samples have been processed.
    #
    # Example log line:
    # {'loss': '1.826', 'grad_norm': '0.4789', 'learning_rate': '3.957e-05', 'entropy': '1.802', 'num_tokens': '2.476e+05',
    #  'mean_token_accuracy': '0.6042', 'epoch': '0.5182'}
    #
    # loss: average training loss over the optimizer steps since the previous log (about the last 16 samples in this setup, assuming 1 GPU).
    # grad_norm: size of the gradient at the current logged optimizer step; useful for monitoring training stability.
    # learning_rate: learning rate used at that point in training.
    # entropy: how uncertain the model's token predictions are; higher means less confident.
    # num_tokens: total number of tokens processed so far.
    # mean_token_accuracy: fraction of tokens predicted correctly in the logged batches.
    # epoch: fraction of the full training epoch completed; 0.5182 means about 51.82%.
    
    sft_trainer = SFTTrainer(
        model = model,
        args = sft_config,
        train_dataset = train_dataset,
        processing_class = tokenizer,
        formatting_func = fchat_example(tokenizer),
    )
    sft_trainer.train()
    
    save_finetuned_model(sft_trainer.model, tokenizer, args.model_save)
    print(f"Saved fine-tuned model to {args.model_save}")

    
def test(args):
    """Test a Qwen3 large language model."""
    print("=" * 60)
    print("TESTING")
    print("=" * 60)

    USE_GPU = torch.cuda.is_available()
    print(f"Using GPU: {USE_GPU}")
    print(f"Test model path: {args.model_load}")

    questions = [
        "Give me an 1-sentence introduction of LLM.",
        "Calculate 1+1-1",
        "What's the difference between thread and process?",
    ]

    model, tokenizer = load_model_and_tokenizer(args.model_load, USE_GPU)
    test_model_with_questions(model, tokenizer, questions)
    del model, tokenizer
    

# ============================================================================
# main()
# ============================================================================

def main():
    """Parse command line arguments and dispatch to train/use/evaluate/prompt."""

    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description='Supervised Fine-Tuning of Qwen LLM.')

    # Mode of operation (mutually exclusive).
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('-train', action='store_true', help='Fine-tune a Qwen model.')
    mode.add_argument('-test', action='store_true', help='Test a Qwen model.')

    # File paths.
    parser.add_argument('-model_load', type=str,
                        default = '/projects/class/itcs6101_091/hw06/models/Qwen3-0.6B-Base',
                        help = 'Path to Qwen model to load.')
    parser.add_argument('-model_save', type=str,
                        default = '../models/Qwen3-0.6B-Base-SFT-qa',
                        help = 'Path to Qwen model to save.')
    parser.add_argument('-train_data', type=str,
                        default = '../data/sft-qa-dataset.json',
                        help = 'Path to the QA training JSON file.')
    
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.test:
        test(args)


if __name__ == '__main__':
    main()
