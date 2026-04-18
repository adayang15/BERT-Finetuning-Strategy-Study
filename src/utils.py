"""Shared utility functions: config loading, logging, seeding, timing, and result saving."""

import os
import json
import time
import random
import logging
import csv
import yaml
import numpy as np
import torch


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, base_path: str = None) -> dict:
    """Load a YAML config file and merge with base.yaml if provided."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if base_path and os.path.exists(base_path):
        with open(base_path) as f:
            base = yaml.safe_load(f)
        config = deep_merge(base, config)

    return config


def setup_logging(experiment_name: str, results_dir: str) -> logging.Logger:
    """Set up console + file logging for an experiment."""
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"{experiment_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    return logging.getLogger(experiment_name)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across random, numpy, torch, and cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> dict:
    """Count total, trainable, and frozen parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": round(100 * trainable / total, 2),
    }


class Timer:
    """Context manager for timing code blocks."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = round(time.time() - self.start, 2)


def save_results(results: dict, path: str):
    """Save results dict to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def save_training_log(history: list, path: str):
    """Save per-epoch training history to a CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
