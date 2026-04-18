"""CLI: Run data efficiency ablation study across all 5 strategies."""

import os
import sys
import json
import argparse
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import load_config, setup_logging, set_seed, count_parameters, save_results
from src.data import get_dataloaders
from src.models import build_model
from src.trainer import Trainer

CONFIGS = [
    "configs/full_finetune.yaml",
    "configs/frozen_6_layers.yaml",
    "configs/frozen_10_layers.yaml",
    "configs/lora_r8.yaml",
    "configs/lora_r16.yaml",
]

FRACTIONS = [0.1, 0.5, 1.0]
ABLATION_EPOCHS = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Data efficiency ablation across all strategies")
    parser.add_argument("--base",    default="configs/base.yaml")
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="figures")
    return parser.parse_args()


def main():
    args   = parse_args()
    logger = setup_logging("ablation", args.results)
    os.makedirs(args.figures, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}

    for cfg_path in CONFIGS:
        config = load_config(cfg_path, args.base)
        config["training"]["num_epochs"] = ABLATION_EPOCHS
        exp    = config["experiment_name"]
        set_seed(config["training"]["seed"])

        all_results[exp] = {}
        for frac in FRACTIONS:
            logger.info(f"Running {exp} | data={frac*100:.0f}%")
            train_loader, val_loader = get_dataloaders(config, data_fraction=frac)
            model   = build_model(config)
            trainer = Trainer(model, config, device)
            trainer.train(train_loader, val_loader)
            metrics = trainer.evaluate(val_loader)
            all_results[exp][f"{int(frac*100)}pct"] = metrics["accuracy"]
            logger.info(f"{exp} | {frac*100:.0f}% data | acc={metrics['accuracy']}")

    save_results(all_results, os.path.join(args.results, "ablation_data_efficiency.json"))

    # Plot data efficiency curves
    plt.figure(figsize=(10, 5))
    x = [10, 50, 100]
    for exp, fracs in all_results.items():
        y = [fracs["10pct"], fracs["50pct"], fracs["100pct"]]
        plt.plot(x, y, marker="o", label=exp)
    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Validation Accuracy")
    plt.title("Data Efficiency: Accuracy vs Training Data Size")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures, "data_efficiency.png"), dpi=150)
    plt.close()

    logger.info("Ablation complete. Results and figures saved.")


if __name__ == "__main__":
    main()
