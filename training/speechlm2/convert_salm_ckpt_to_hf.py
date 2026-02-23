import argparse
from pathlib import Path

import torch
from huggingface_hub import add_collection_item, metadata_update
from nemo.collections.speechlm2 import SALM
from omegaconf import OmegaConf

from utils.logger import Logger


def _find_best_checkpoint(ckpt_dir: str) -> str:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise ValueError(f"{ckpt_dir} is not a directory")
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in {ckpt_dir}")
    best_ckpt = min(ckpt_files, key=lambda x: float(x.stem.split("val_acc=")[-1]))
    return str(best_ckpt)

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    ckpt_data = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt_data["state_dict"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to experiment config (exp_config.yaml from NeMo logs directory)')
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--prompt_format', type=str, required=True)
    parser.add_argument('--model_name_for_hf_upload', type=str, required=False)
    parser.add_argument('--tags', type=str, nargs='*', default=[],
                        help='Tags for HuggingFace model (e.g., --tags 2026-02-23 librispeech v1)')
    parser.add_argument('--collection_slug', type=str, required=False,
                        help='HuggingFace collection slug to add model to')
    args = parser.parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(args)
    Logger.info("===================================")

    Logger.info("Creating model config ...")
    Logger.info(f"Loading config from: {args.config_path}")
    training_cfg = OmegaConf.load(args.config_path)
    model_cfg = OmegaConf.to_container(training_cfg.model, resolve=True)
    model_cfg["torch_dtype"] = "bfloat16"
    model_cfg["pretrained_llm"] = args.llm_name
    model_cfg["prompt_format"] = args.prompt_format

    Logger.info("Loading model ...")
    model = SALM(model_cfg)

    Logger.info("Finding best checkpoint ...")
    best_checkpoint_path = _find_best_checkpoint(args.ckpt_dir)
    Logger.info(f"Found best checkpoint: {best_checkpoint_path}")

    Logger.info("Loading checkpoint ...")
    load_checkpoint(model, best_checkpoint_path)
    model = model.to(getattr(torch, "bfloat16"))

    Logger.info("Saving model in HF format ...")
    hf_save_dir = Path(args.ckpt_dir) / "hf"
    hf_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_save_dir)
    Logger.info(f"Model saved to {hf_save_dir}")
    if args.model_name_for_hf_upload:
        Logger.info("Pushing model to HuggingFace Hub ...")
        repo_id = f"aklemen/{args.model_name_for_hf_upload}"
        model.push_to_hub(repo_id)
        Logger.info(f"Model pushed to HuggingFace Hub as {repo_id}")

        if args.tags:
            Logger.info(f"Adding tags: {args.tags}")
            metadata_update(repo_id=repo_id, metadata={"tags": args.tags})

        if args.collection_slug:
            Logger.info(f"Adding to collection: {args.collection_slug}")
            add_collection_item(
                collection_slug=args.collection_slug,
                item_id=repo_id,
                item_type="model"
            )


if __name__ == "__main__":
    main()
