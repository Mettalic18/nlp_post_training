#!/usr/bin/env python
# coding: utf-8

import os
import re
import argparse

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


SYSTEM_MESSAGE = (
    "You are a careful math tutor. Solve the problem step by step. "
    "Finish with a final line in the exact format `#### answer`."
)


def ensure_chat_template(tokenizer):
    if getattr(tokenizer, "chat_template", None):
        return

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


def generate_responses(model, tokenizer, user_message, system_message=None, max_new_tokens=160):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    ensure_chat_template(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32,
    )

    if use_gpu:
        model.to("cuda")

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.generation_config.do_sample = False
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)

    return model, tokenizer


def save_finetuned_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def extract_message_text(item):
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        return str(item.get("content", "")).strip()

    if isinstance(item, list):
        for message in reversed(item):
            if isinstance(message, dict) and "content" in message:
                return str(message["content"]).strip()
            if isinstance(message, str):
                return message.strip()

    return str(item).strip()


def normalize_answer(text):
    if text is None:
        return None

    text = str(text).strip()
    if not text:
        return None

    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = re.sub(r"\s+", "", text)
    return text or None


def extract_ground_truth(answer_text):
    match = re.search(r"####\s*(.+)", str(answer_text))
    if match:
        return normalize_answer(match.group(1))
    return normalize_answer(answer_text)


def extract_model_answer(completion_text):
    completion_text = extract_message_text(completion_text)

    match = re.search(r"####\s*([^\n]+)", completion_text)
    if match:
        return normalize_answer(match.group(1))

    matches = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", completion_text)
    if matches:
        return normalize_answer(matches[-1])

    return None


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = extract_message_text(completion)
        has_format = re.search(r"####\s*([^\n]+)", text) is not None
        rewards.append(0.5 if has_format else 0.0)
    return rewards


def correctness_reward(completions, ground_truth, **kwargs):
    rewards = []
    for completion, target in zip(completions, ground_truth):
        predicted = extract_model_answer(completion)
        expected = normalize_answer(target)
        rewards.append(1.0 if predicted is not None and predicted == expected else 0.0)
    return rewards


def build_prompt(question):
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": str(question).strip()},
    ]


def build_gsm_record(question, answer):
    ground_truth = extract_ground_truth(answer)
    if ground_truth is None:
        return None

    return {
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "prompt": build_prompt(question),
        "ground_truth": ground_truth,
    }


def prepare_gsm_dataset(dataset):
    columns = set(dataset.column_names)
    if "prompt" in columns and "ground_truth" in columns:
        return dataset

    if not {"question", "answer"}.issubset(columns):
        raise ValueError(
            "Training data must contain either `prompt` and `ground_truth`, "
            "or raw GSM fields `question` and `answer`."
        )

    dataset = dataset.map(
        lambda example: {
            "prompt": build_prompt(example["question"]),
            "ground_truth": extract_ground_truth(example["answer"]),
        }
    )
    dataset = dataset.filter(lambda example: example["ground_truth"] is not None)
    return dataset

def load_training_dataset(args):
    raw_dataset = load_dataset("json", data_files=args.train_data)["train"]

    if args.max_train_samples > 0 and len(raw_dataset) > args.max_train_samples:
        raw_dataset = raw_dataset.select(range(args.max_train_samples))

    processed_dataset = prepare_gsm_dataset(raw_dataset)
    return raw_dataset, processed_dataset


def train(args):
    """Fine-tune a Qwen large language model with GRPO on GSM-style questions."""
    print("=" * 60)
    print("FINE-TUNING")
    print("=" * 60)

    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}")
    print(f"Base model path: {args.model_load}")
    print(f"GRPO model path: {args.model_save}")
    print(f"Training data source: {args.train_data}")

    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if effective_batch_size % args.num_generations != 0:
        raise ValueError(
            "per_device_train_batch_size * gradient_accumulation_steps must be "
            "divisible by num_generations for this single-process setup."
        )

    raw_dataset, train_dataset = load_training_dataset(args)

    print("Raw dataset example:")
    print(raw_dataset[0])
    print("\nFormatted GRPO example:")
    print(train_dataset[0])
    print(f"\nNumber of training samples: {len(train_dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model_load, use_gpu=False)

    grpo_config = GRPOConfig(
        output_dir=args.model_save,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        logging_steps=2,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
    )
    # In this homework setup, each prompt produces several sampled completions.
    # GRPO then compares rewards within each prompt group instead of requiring a separate reward model.
    #
    # With the defaults below on one process:
    # effective batch size = per_device_train_batch_size * gradient_accumulation_steps = 1 * 8 = 8 prompts
    # num_generations = 4 means each prompt is expanded into 4 sampled completions
    # logging_steps = 2 prints metrics every 2 optimizer steps

    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=[format_reward, correctness_reward],
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    grpo_trainer.train()

    save_finetuned_model(grpo_trainer.model, tokenizer, args.model_save)
    print(f"Saved fine-tuned model to {args.model_save}")


def test(args):
    """Test a GRPO-tuned Qwen model on GSM-style questions."""
    print("=" * 60)
    print("TESTING")
    print("=" * 60)

    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}")
    print(f"Test model path: {args.model_load}")

    questions = [
        "Janet has 3 packs of pencils with 4 pencils in each pack. She buys 5 more pencils. How many pencils does she have now?",
        "A bakery sells 18 muffins in the morning and 27 muffins in the afternoon. If each muffin costs 2 dollars, how much money did the bakery make from muffins that day?",
        "Tom read 12 pages on Monday, 15 pages on Tuesday, and 9 pages on Wednesday. His book has 50 pages. How many pages does he still need to read?",
    ]

    model, tokenizer = load_model_and_tokenizer(args.model_load, use_gpu)
    test_model_with_questions(model, tokenizer, questions, SYSTEM_MESSAGE)
    del model, tokenizer


def main():
    """Parse command line arguments and dispatch to train or test."""

    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Group Relative Policy Optimization of Qwen LLM on GSM."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-train", action="store_true", help="Fine-tune a Qwen model with GRPO.")
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
        default="../models/Qwen2.5-0.5B-Instruct-GRPO-gsm",
        help="Path to Qwen model to save.",
    )
    parser.add_argument(
        "-train_data",
        type=str,
        default="../data/grpo-gsm-dataset.json",
        help="Path to the GRPO training JSON file.",
    )
    parser.add_argument(
        "-max_train_samples",
        type=int,
        default=500,
        help="Maximum number of training samples to use. Use -1 for the full split.",
    )
    parser.add_argument(
        "-learning_rate",
        type=float,
        default=1e-6,
        help="GRPO learning rate.",
    )
    parser.add_argument(
        "-num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "-per_device_train_batch_size",
        type=int,
        default=1,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "-gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "-num_generations",
        type=int,
        default=4,
        help="Number of sampled completions per prompt for GRPO.",
    )
    parser.add_argument(
        "-max_prompt_length",
        type=int,
        default=256,
        help="Maximum prompt length in tokens.",
    )
    parser.add_argument(
        "-max_completion_length",
        type=int,
        default=160,
        help="Maximum completion length in tokens.",
    )
    parser.add_argument(
        "-temperature",
        type=float,
        default=0.9,
        help="Sampling temperature used during GRPO rollouts.",
    )

    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.test:
        test(args)


if __name__ == "__main__":
    main()
