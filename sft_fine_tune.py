import argparse
import os

import torch
from accelerate import PartialState
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from logger import Logger

os.environ["WANDB_PROJECT"] = "H2T-LoRA"

Logger.info(f"Using device map: {device_map}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=False)
    parser.add_argument('--manifest_file_path', type=str, required=True)
    parser.add_argument('--beams_file_path', type=str, required=True)
    parser.add_argument('--beam_size', type=int, required=True)
    parser.add_argument('--output_dir_path', type=str, required=True)
    parser.add_argument('--tokenized_dataset_dir_path', type=str, required=True)
    parser.add_argument("--run_name", type=str, default="lora-finetune")

    parser.add_argument("--lora_r", type=float, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--per_device_batch_size", type=int, default=8)

    arguments = parser.parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(arguments)
    Logger.info("===================================")
    return arguments


response_template = "### Transkript:"

def main():
    args = parse_args()

    if args.tokenizer_name is None:
        Logger.info(f"Tokenizer name was not given, using LLM name '{args.llm_name}'")
        args.tokenizer_name = args.llm_name

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        Logger.info(f"No pad_token available. Setting pad_token to eos_token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    Logger.info(f"Loading tokenized datasets (train, val) from {args.tokenized_dataset_dir_path} ...")
    tokenized_train_path = os.path.join(args.tokenized_dataset_dir_path, "train")
    tokenized_val_path = os.path.join(args.tokenized_dataset_dir_path, "val")
    tokenized_train = load_from_disk(tokenized_train_path)
    tokenized_val = load_from_disk(tokenized_val_path)

    Logger.info(f"Train dataset: {tokenized_train}")
    Logger.info(f"Val dataset: {tokenized_val}")

    llm = AutoModelForCausalLM.from_pretrained(
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

    Logger.info(f"Model BEFORE applying LoRA: {llm}")
    llm = get_peft_model(llm, peft_config)
    Logger.info(f"Model AFTER applying LoRA: {llm}")

    llm.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir_path,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # prediction_loss_only=True, # Returns only the loss, no predictions
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
        model=llm,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        max_seq_length=512,
        args=training_args
    )

    trainer.train()

    llm.save_pretrained(f"{args.output_dir_path}/adapter")
    tokenizer.save_pretrained(f"{args.output_dir_path}/tokenizer")


if __name__ == '__main__':
    main()