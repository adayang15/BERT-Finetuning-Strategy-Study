"""CLI: Evaluate all 5 experiments and generate comparison visualizations."""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


EXPERIMENTS = [
    "full_finetune",
    "frozen_6_layers",
    "frozen_10_layers",
    "lora_r8",
    "lora_r16",
]

LABELS = {
    "full_finetune":    "Full Finetune",
    "frozen_6_layers":  "Frozen 6 Layers",
    "frozen_10_layers": "Frozen 10 Layers",
    "lora_r8":          "LoRA r=8",
    "lora_r16":         "LoRA r=16",
}


def load_all_results(results_dir: str) -> dict:
    """Load all experiment JSON result files."""
    data = {}
    for exp in EXPERIMENTS:
        path = os.path.join(results_dir, f"{exp}.json")
        if os.path.exists(path):
            with open(path) as f:
                data[exp] = json.load(f)
    return data


def plot_accuracy_comparison(data: dict, figures_dir: str):
    """Bar chart: accuracy of all 5 models."""
    names = [LABELS[e] for e in data]
    accs  = [data[e]["final"]["accuracy"] for e in data]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, accs, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"])
    plt.ylim(0.8, 1.0)
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy Comparison Across Fine-tuning Strategies")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{acc:.4f}", ha="center", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()


def plot_params_vs_accuracy(data: dict, figures_dir: str):
    """Scatter plot: trainable params (log scale) vs accuracy."""
    plt.figure(figsize=(8, 5))
    for exp, d in data.items():
        params = d["parameters"]["trainable"]
        acc    = d["final"]["accuracy"]
        plt.scatter(params, acc, s=120, label=LABELS[exp], zorder=5)
        plt.annotate(LABELS[exp], (params, acc), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
    plt.xscale("log")
    plt.xlabel("Trainable Parameters (log scale)")
    plt.ylabel("Validation Accuracy")
    plt.title("Parameter Efficiency: Params vs Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "params_vs_accuracy.png"), dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all experiments and generate figures")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--figures", default="figures", help="Figures output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figures, exist_ok=True)

    data = load_all_results(args.results)
    if not data:
        print("No results found. Run train.py for each strategy first.")
        return

    print("\n=== Results Summary ===")
    print(f"{'Strategy':<20} {'Accuracy':>10} {'F1':>10} {'Trainable Params':>18}")
    print("-" * 62)
    for exp, d in data.items():
        print(f"{LABELS[exp]:<20} {d['final']['accuracy']:>10} "
              f"{d['final']['f1']:>10} {d['parameters']['trainable']:>18,}")

    plot_accuracy_comparison(data, args.figures)
    plot_params_vs_accuracy(data, args.figures)
    print(f"\nFigures saved to {args.figures}/")


if __name__ == "__main__":
    main()
