import argparse
import os

import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from logger import Logger
from torch_datasets.manifest_dataset import ManifestDataset

os.environ["WANDB_PROJECT"] = "H2T-LoRA"

response_template = "### Transkript:"
prompt_template = ("### Navodilo:\n"
                   "Spodaj je najboljša hipoteza, ki jo je za avdio posnetek generiral sistem za razpoznavanje govora. "
                   "Preglej jo in jo s pomočjo ostalih hipotez popravi, če je potebno. "
                   "Potem izpiši končni transkript.\n\n"
                   "### Najboljša hipoteza:\n{best_hypothesis}\n\n"
                   "### Ostale hipoteze:\n{other_hypotheses}\n\n"
                   f"{response_template}\n"
                   "{ground_truth}")


def generate_prompt(hypotheses: list[str], ground_truth: str):
    return prompt_template.format_map({
        "best_hypothesis": hypotheses[0],
        "other_hypotheses": "\n".join(hypotheses[1:]),
        "ground_truth": ground_truth,
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=False)
    parser.add_argument('--manifest_file_path', type=str, required=True)
    parser.add_argument('--beams_file_path', type=str, required=True)
    parser.add_argument('--beam_size', type=int, required=True)
    parser.add_argument('--output_dir_path', type=str, required=True)
    parser.add_argument('--tokenized_dataset_dir_path', type=str, required=False)
    parser.add_argument("--run_name", type=str, default="lora-finetune")

    parser.add_argument("--lora_r", type=float, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--per_device_batch_size", type=int, default=8)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(args)
    Logger.info("===================================")

    if args.tokenizer_name is None:
        Logger.info(f"Tokenizer name was not given, using LLM name '{args.llm_name}'")
        args.tokenizer_name = args.llm_name

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        Logger.info(f"No pad_token available. Setting pad_token to eos_token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.llm_name,
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    def tokenize(examples):
        prompts = [f"{generate_prompt(h, g)}{tokenizer.eos_token}" for h, g in
                   zip(examples["hypotheses"], examples["ground_truth"])]

        return tokenizer(
            prompts,
            padding=True,
            padding_side="right",
            add_special_tokens=False
        )

    tokenized_train_path = os.path.join(args.tokenized_dataset_dir_path, "train")
    tokenized_val_path = os.path.join(args.tokenized_dataset_dir_path, "val")
    if os.path.exists(tokenized_train_path) and os.path.exists(tokenized_val_path):
        Logger.info(f"Loading tokenized datasets (train, val) from {args.tokenized_dataset_dir_path} ...")
        tokenized_train = load_from_disk(tokenized_train_path)
        tokenized_val = load_from_disk(tokenized_val_path)
        Logger.info(f"Train and val datasets loaded.")
    else:
        Logger.info(f"Loading ground truths from {args.manifest_file_path} ...")
        manifest = ManifestDataset(args.manifest_file_path)
        ground_truths = manifest.get_transcripts()

        Logger.info(f"Loading hypotheses from {args.beams_file_path} ...")
        hypotheses = pd.read_csv(args.beams_file_path, delimiter="\t", header=None, names=["text", "score"])
        hypotheses = hypotheses["text"].tolist()
        grouped_hypotheses = [hypotheses[i:i + args.beam_size] for i in range(0, len(hypotheses), args.beam_size)]

        Logger.info(f"Creating and splitting dataset ...")
        dataset = Dataset.from_dict({
            "hypotheses": grouped_hypotheses,
            "ground_truth": ground_truths,
        })
        train_val = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

        Logger.info("Tokenizing dataset ...")
        tokenized_train = train_val["train"].map(tokenize, batched=True, remove_columns=["hypotheses", "ground_truth"])
        tokenized_val = train_val["test"].map(tokenize, batched=True, remove_columns=["hypotheses", "ground_truth"])
        Logger.info(f"Tokenization complete.")
        if args.tokenized_dataset_dir_path is not None:
            Logger.info(f"Saving tokenized datasets (train, val) to {args.tokenized_dataset_dir_path} ...")
            os.makedirs(args.tokenized_dataset_dir_path,  exist_ok=True)
            tokenized_train.save_to_disk(tokenized_train_path)
            tokenized_val.save_to_disk(tokenized_val_path)

    Logger.info(f"Train dataset: {tokenized_train}")
    Logger.info(f"Val dataset: {tokenized_val}")

    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

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
        num_train_epochs=5,
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
        # gradient_accumulation_steps=8,
        optim="adamw_torch",
        learning_rate=1e-4,
        warmup_ratio=0.05,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-5},
        bf16=True,
        dataloader_num_workers=8,
        push_to_hub=False
    )

    trainer = Trainer(
        model=llm,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    trainer.train()

    llm.save_pretrained(f"{args.output_dir_path}/adapter")
    tokenizer.save_pretrained(f"{args.output_dir_path}/tokenizer")

