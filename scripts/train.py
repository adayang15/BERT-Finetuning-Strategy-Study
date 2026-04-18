"""CLI entry point for training a single BERT fine-tuning strategy."""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import load_config, setup_logging, set_seed, count_parameters, save_results, save_training_log
from src.data import get_dataloaders
from src.models import build_model
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BERT fine-tuning strategy on SST-2")
    parser.add_argument("--config",   required=True, help="Path to strategy config yaml")
    parser.add_argument("--base",     default="configs/base.yaml", help="Path to base config yaml")
    parser.add_argument("--results",  default="results", help="Directory to save results")
    parser.add_argument("--figures",  default="figures", help="Directory to save figures")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config, args.base)

    experiment = config["experiment_name"]
    logger     = setup_logging(experiment, args.results)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Load data
    train_loader, val_loader = get_dataloaders(config)

    # 2. Build model
    model  = build_model(config)
    params = count_parameters(model)
    logger.info(f"Parameters: {params}")

    # 3. Train
    trainer = Trainer(model, config, device)
    history = trainer.train(train_loader, val_loader)

    # 4. Final evaluation
    final_metrics = trainer.evaluate(val_loader)
    logger.info(f"Final | accuracy={final_metrics['accuracy']} | f1={final_metrics['f1']}")

    # 5. Save results
    results = {
        "experiment": experiment,
        "strategy":   config["strategy"],
        "parameters": params,
        "final":      final_metrics,
        "history":    history,
    }
    save_results(results,      os.path.join(args.results, f"{experiment}.json"))
    save_training_log(history, os.path.join(args.results, f"{experiment}_log.csv"))
    logger.info(f"Results saved to {args.results}/")


if __name__ == "__main__":
    main()
