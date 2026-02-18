"""
Callback to log trainable and total parameters to WandB during training.
"""
from collections import defaultdict
from typing import Dict, Tuple

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class LogTrainableParamsCallback(Callback):
    """
    PyTorch Lightning callback that counts and logs trainable vs. total parameters
    to WandB, both overall and broken down by module.
    """

    def __init__(self, log_by_module: bool = True):
        """
        Args:
            log_by_module: If True, also log parameter counts broken down by top-level modules
        """
        super().__init__()
        self.log_by_module = log_by_module

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when training starts. Counts parameters and logs to WandB.
        """
        # Count overall parameters
        total_params, trainable_params = self._count_parameters(pl_module)
        frozen_params = total_params - trainable_params
        trainable_percentage = (trainable_params / total_params * 100) if total_params > 0 else 0

        # Log overall metrics
        metrics = {
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "model/frozen_params": frozen_params,
            "model/trainable_percentage": trainable_percentage,
        }

        # Count and log by module if requested
        module_stats = None
        if self.log_by_module:
            module_stats = self._count_parameters_by_module(pl_module)
            for module_name, (total, trainable) in module_stats.items():
                frozen = total - trainable
                percentage = (trainable / total * 100) if total > 0 else 0
                metrics[f"model/{module_name}/total_params"] = total
                metrics[f"model/{module_name}/trainable_params"] = trainable
                metrics[f"model/{module_name}/frozen_params"] = frozen
                metrics[f"model/{module_name}/trainable_percentage"] = percentage

        # Log to WandB via trainer's logger
        if trainer.logger is not None:
            # For WandB logger, log as summary metrics
            if hasattr(trainer.logger, 'experiment'):
                for key, value in metrics.items():
                    trainer.logger.experiment.summary[key] = value
            # Also log as regular metrics (will appear in history)
            trainer.logger.log_metrics(metrics, step=0)

        # Print summary to console
        self._print_summary(total_params, trainable_params, frozen_params, trainable_percentage, module_stats if self.log_by_module else None)

    def _count_parameters(self, module: pl.LightningModule) -> Tuple[int, int]:
        """
        Count total and trainable parameters in the model.

        Args:
            module: PyTorch Lightning module

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total_params, trainable_params

    def _count_parameters_by_module(self, module: pl.LightningModule) -> Dict[str, Tuple[int, int]]:
        """
        Count parameters by top-level module.

        Args:
            module: PyTorch Lightning module

        Returns:
            Dictionary mapping module names to (total_params, trainable_params) tuples
        """
        module_stats = defaultdict(lambda: [0, 0])  # [total, trainable]

        for name, param in module.named_parameters():
            # Get top-level module name (first part before the dot)
            top_level_name = name.split('.')[0] if '.' in name else name

            module_stats[top_level_name][0] += param.numel()  # total
            if param.requires_grad:
                module_stats[top_level_name][1] += param.numel()  # trainable

        # Convert to dict of tuples
        return {name: (total, trainable) for name, (total, trainable) in module_stats.items()}

    @rank_zero_only
    def _print_summary(
        self,
        total_params: int,
        trainable_params: int,
        frozen_params: int,
        trainable_percentage: float,
        module_stats: Dict[str, Tuple[int, int]] = None
    ) -> None:
        """
        Print a formatted summary of parameter counts.
        """
        print("\n" + "=" * 70)
        print("TRAINABLE PARAMETERS SUMMARY")
        print("=" * 70)
        print(f"Total parameters:      {total_params:,}")
        print(f"Trainable parameters:  {trainable_params:,} ({trainable_percentage:.2f}%)")
        print(f"Frozen parameters:     {frozen_params:,} ({100 - trainable_percentage:.2f}%)")

        if module_stats:
            print("\n" + "-" * 70)
            print("BY MODULE:")
            print("-" * 70)
            # Sort by module name for consistent output
            for module_name in sorted(module_stats.keys()):
                total, trainable = module_stats[module_name]
                frozen = total - trainable
                percentage = (trainable / total * 100) if total > 0 else 0
                print(f"\n{module_name}:")
                print(f"  Total:      {total:,}")
                print(f"  Trainable:  {trainable:,} ({percentage:.2f}%)")
                print(f"  Frozen:     {frozen:,} ({100 - percentage:.2f}%)")

        print("=" * 70 + "\n")


