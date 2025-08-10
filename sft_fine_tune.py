import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from logger import Logger

os.environ["WANDB_PROJECT"] = "H2T-LoRA"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=False)
    parser.add_argument('--prompt_completion_dataset_name', type=str, required=True)
    parser.add_argument('--output_dir_path', type=str, required=True)
    parser.add_argument("--run_name", type=str, default="lora-finetune")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--num_samples", type=int, required=False, help="Number of samples to use from the dataset for testing purposes.")

    arguments = parser.parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(arguments)
    Logger.info("===================================")
    return arguments


def convert_to_standard_format(example):
    return {
        "prompt": example["prompt"][0]["content"],
        "completion": example["completion"][0]["content"],
    }


def main():
    args = parse_args()

    if args.tokenizer_name is None:
        Logger.info(f"Tokenizer name was not given, using LLM name '{args.llm_name}'")
        args.tokenizer_name = args.llm_name

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        Logger.info(f"No pad_token available. Setting pad_token to eos_token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.prompt_completion_dataset_name)['train']
    if args.num_samples is not None:
        Logger.warn(f"Only using {args.num_samples} samples from the dataset. Only use this for testing purposes!")
        dataset = dataset.select(range(args.num_samples))
    dataset = dataset.map(convert_to_standard_format)
    Logger.info(f"Example from dataset: {dataset[0]}")
    train_val_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    Logger.info(f"Dataset 80/20 split: {train_val_dataset}")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.llm_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
    )

    os.makedirs(args.output_dir_path, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=args.output_dir_path,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{args.output_dir_path}/logs",
        logging_strategy="steps",
        logging_steps=1000,
        logging_first_step=True,
        report_to="wandb",
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=8,
        packing=True,
        max_length=512,
        optim="adamw_torch",
        learning_rate=5e-5,
        warmup_ratio=0.05,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-7},
        bf16=True,
        dataloader_num_workers=8,
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=train_val_dataset['train'],
        eval_dataset=train_val_dataset['test'],
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train()

    if trainer.is_world_process_zero():
        Logger.info("Training completed. Saving model and tokenizer...")
        trainer.model.save_pretrained(f"{args.output_dir_path}/adapter")
        tokenizer.save_pretrained(f"{args.output_dir_path}/tokenizer")

        Logger.info("Merging and pushing model to Hugging Face Hub...")
        merged_model = trainer.model.merge_and_unload()
        was_test_run = args.num_samples is not None
        model_name = "H2T-LoRA-test" if was_test_run else "H2T-LoRA"
        merged_model.push_to_hub(f"aklemen/{model_name}")


if __name__ == '__main__':
    main()
