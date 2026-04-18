"""Training and evaluation engine for BERT fine-tuning experiments."""

import logging
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from src.utils import Timer

logger = logging.getLogger(__name__)


class Trainer:
    """Handles training loop, evaluation, and metric collection."""

    def __init__(self, model, config: dict, device: torch.device):
        self.model  = model.to(device)
        self.config = config
        self.device = device
        self.history = []

    def _build_optimizer_and_scheduler(self, num_training_steps: int):
        """Create AdamW optimizer and linear warmup scheduler."""
        lr           = self.config["strategy"]["learning_rate"]
        weight_decay = self.config["training"]["weight_decay"]
        warmup_ratio = self.config["training"]["warmup_ratio"]

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    def train(self, train_loader, val_loader):
        """Run full training across all epochs and return history."""
        num_epochs         = self.config["training"]["num_epochs"]
        max_grad_norm      = self.config["training"]["max_grad_norm"]
        num_training_steps = num_epochs * len(train_loader)

        optimizer, scheduler = self._build_optimizer_and_scheduler(num_training_steps)

        logger.info(f"Starting training | epochs={num_epochs} | steps={num_training_steps}")

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0.0

            with Timer() as t:
                for batch in train_loader:
                    batch     = {k: v.to(self.device) for k, v in batch.items()}
                    outputs   = self.model(**batch)
                    loss      = outputs.loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()

            avg_loss = round(total_loss / len(train_loader), 4)
            val_metrics = self.evaluate(val_loader)

            epoch_log = {
                "epoch":       epoch,
                "train_loss":  avg_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1":      val_metrics["f1"],
                "epoch_time_s": t.elapsed,
            }
            self.history.append(epoch_log)
            logger.info(
                f"Epoch {epoch}/{num_epochs} | loss={avg_loss} | "
                f"acc={val_metrics['accuracy']} | f1={val_metrics['f1']} | "
                f"time={t.elapsed}s"
            )

        return self.history

    def evaluate(self, val_loader) -> dict:
        """Evaluate model on validation set and return accuracy and macro F1."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch   = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds   = outputs.logits.argmax(dim=-1).cpu().tolist()
                labels  = batch["labels"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)

        return {
            "accuracy": round(accuracy_score(all_labels, all_preds), 4),
            "f1":       round(f1_score(all_labels, all_preds, average="macro"), 4),
        }
