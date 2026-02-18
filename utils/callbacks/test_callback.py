"""
Test script to verify the LogTrainableParamsCallback works correctly.
This creates a simple model with frozen and trainable parameters.
"""
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = nn.Linear(100, 50)
        self.perception = nn.Sequential(
            nn.Linear(50, 30),
            nn.Linear(30, 20)
        )
        self.embed_tokens = nn.Embedding(1000, 100)

        # Freeze LLM and embed_tokens (similar to SALM config)
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.perception(self.llm(x))


def test_parameter_counting():
    """Test the parameter counting logic from the callback."""
    from collections import defaultdict

    model = SimpleModel()

    # Count overall parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("=" * 70)
    print("PARAMETER COUNTING TEST")
    print("=" * 70)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Frozen parameters:     {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")

    # Count by module
    module_stats = defaultdict(lambda: [0, 0])
    for name, param in model.named_parameters():
        top_level_name = name.split('.')[0] if '.' in name else name
        module_stats[top_level_name][0] += param.numel()
        if param.requires_grad:
            module_stats[top_level_name][1] += param.numel()

    print("\n" + "-" * 70)
    print("BY MODULE:")
    print("-" * 70)
    for module_name in sorted(module_stats.keys()):
        total, trainable = module_stats[module_name]
        frozen = total - trainable
        percentage = (trainable / total * 100) if total > 0 else 0
        print(f"\n{module_name}:")
        print(f"  Total:      {total:,}")
        print(f"  Trainable:  {trainable:,} ({percentage:.2f}%)")
        print(f"  Frozen:     {frozen:,} ({100 - percentage:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    test_parameter_counting()

