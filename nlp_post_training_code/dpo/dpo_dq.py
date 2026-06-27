#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


def generate_responses(model, tokenizer, user_message, system_message=None, max_new_tokens=128):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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


def test_model_with_questions(model, tokenizer, questions, system_message=None):
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


def display_dataset(dataset):
    rows = []
    for i in range(min(3, len(dataset))):
        rows.append({
            "Prompt": dataset[i]["chosen"][-2]["content"],
            "Chosen Response": dataset[i]["chosen"][-1]["content"],
            "Rejected Response": dataset[i]["rejected"][-1]["content"],
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", None)
    print(df.to_string(index=False))


def save_finetuned_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))


def train(args):
    """Fine-tune a Qwen2.5 large language model with DPO."""
    print("=" * 60)
    print("FINE-TUNING")
    print("=" * 60)

    USE_GPU = torch.cuda.is_available()

    print(f"Using GPU: {USE_GPU}")
    print(f"Base model path: {args.model_load}")
    print(f"DPO model path: {args.model_save}")

    model, tokenizer = load_model_and_tokenizer(args.model_load, USE_GPU)

    train_dataset = load_dataset("json", data_files=args.train_data)["train"]
    if not USE_GPU:
        train_dataset = train_dataset.select(range(500))

    print("Raw dataset example:")
    print(train_dataset[0])
    print("\nPreview of chosen and rejected responses:")
    display_dataset(train_dataset)

    dpo_config = DPOConfig(
        output_dir=args.model_save,
        beta=0.2,
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        logging_steps=2,
        save_strategy="no",
        report_to="none",
    )
    # Here, 1 sample is loaded at a time on GPU.
    # Gradients are accumulated for 8 forward/backward passes before one optimizer update.
    # So the effective batch size per optimizer step is: per_device_train_batch_size * gradient_accumulation_steps = 1 * 8 = 8 samples.
    #
    # The training split has 1000 samples in this homework setup when using GPU,
    # so one epoch takes about: 1000 / 8 = 125 optimizer steps.
    #
    # logging_steps = 2 means training metrics are printed every 2 optimizer steps.
    # Since each optimizer step corresponds to 8 samples, logs appear after about: 2 * 8 = 16 samples have been processed.
    #
    # Example log line:
    # {'loss': '0.611', 'grad_norm': '6.812', 'learning_rate': '4.92e-06', 'rewards/chosen': '0.245',
    #  'rewards/rejected': '-0.141', 'rewards/accuracies': '0.750', 'rewards/margins': '0.386', 'epoch': '0.032'}
    #
    # loss: average DPO training loss over the optimizer steps since the previous log.
    # grad_norm: size of the gradient at the current logged optimizer step; useful for monitoring training stability.
    # learning_rate: learning rate used at that point in training.
    # rewards/chosen: average reward assigned to the preferred responses in the logged steps.
    # rewards/rejected: average reward assigned to the rejected responses in the logged steps.
    # rewards/accuracies: fraction of comparisons where the chosen response received a higher reward than the rejected response.
    # rewards/margins: average difference between chosen reward and rejected reward.
    # epoch: fraction of the full training epoch completed.

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )
    dpo_trainer.train()

    save_finetuned_model(dpo_trainer.model, tokenizer, args.model_save)
    print(f"Saved fine-tuned model to {args.model_save}")


def test(args):
    """Test a Qwen2.5 large language model."""
    print("=" * 60)
    print("TESTING")
    print("=" * 60)

    USE_GPU = torch.cuda.is_available()
    print(f"Using GPU: {USE_GPU}")
    print(f"Test model path: {args.model_load}")

    questions = [
        "What is your name?",
        "Are you ChatGPT?",
        "Tell me about your name and organization.",
    ]

    model, tokenizer = load_model_and_tokenizer(args.model_load, USE_GPU)
    test_model_with_questions(model, tokenizer, questions)
    del model, tokenizer


def main():
    """Parse command line arguments and dispatch to train or test."""

    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Direct Preference Optimization of Qwen LLM."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-train", action="store_true", help="Fine-tune a Qwen model with DPO.")
    mode.add_argument("-test", action="store_true", help="Test a Qwen model.")

    parser.add_argument(
        "-model_load",
        type=str,
        default="/projects/class/itcs6101_091/hw06/models/Qwen2.5-0.5B-Instruct",
        help="Path to Qwen model to load.",
    )
    parser.add_argument(
        "-model_save",
        type=str,
        default="../models/Qwen2.5-0.5B-Instruct-DPO-dq",
        help="Path to Qwen model to save.",
    )
    parser.add_argument(
        "-train_data",
        type=str,
        default="../data/dpo-dq-dataset.json",
        help="Path to the DPO training JSON file.",
    )

    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.test:
        test(args)


if __name__ == "__main__":
    main()
