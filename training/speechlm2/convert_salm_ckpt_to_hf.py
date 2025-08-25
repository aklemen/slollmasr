import argparse
from pathlib import Path

import torch
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
    best_ckpt = min(ckpt_files, key=lambda x: float(x.stem.split("val_loss=")[-1]))
    return str(best_ckpt)

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    ckpt_data = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt_data["state_dict"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', nargs='+', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--upload_to_hub', action='store_true')
    args = parser.parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(args)
    Logger.info("===================================")

    Logger.info("Creating model config ...")
    training_cfg = OmegaConf.load("/slollmasr/training/speechlm2/salm.yaml")
    model_cfg = OmegaConf.to_container(training_cfg.model, resolve=True)
    model_cfg["torch_dtype"] = "bfloat16"

    Logger.info("Loading model ...")
    model = SALM(model_cfg)

    Logger.info("Finding best checkpoint ...")
    best_checkpoint_path = _find_best_checkpoint(args.ckpt_dir)
    Logger.info(f"Found best checkpoint: {best_checkpoint_path}")

    Logger.info("Loading checkpoint ...")
    load_checkpoint(model, best_checkpoint_path)
    model = model.to(getattr(torch, "bfloat16"))

    Logger.info("Saving model in HF format ...")
    model.save_pretrained(args.ckpt_dir)
    Logger.info(f"Model saved to {args.ckpt_dir}")
    if args.upload_to_hub:
        Logger.info("Pushing model to HuggingFace Hub ...")
        repo_id = f"aklemen/{args.model_name}"
        model.push_to_hub(repo_id)
        Logger.info(f"Model pushed to HuggingFace Hub as {repo_id}")


if __name__ == "__main__":
    main()
