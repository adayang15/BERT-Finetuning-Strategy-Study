"""Data loading and tokenization for SST-2 sentiment analysis."""

import logging
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


def get_tokenizer(pretrained: str = "bert-base-uncased") -> BertTokenizer:
    """Load the BERT tokenizer."""
    return BertTokenizer.from_pretrained(pretrained)


def tokenize_dataset(dataset, tokenizer: BertTokenizer, max_length: int = 128):
    """Tokenize a HuggingFace dataset split with BERT tokenizer."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def get_dataloaders(config: dict, data_fraction: float = 1.0):
    """
    Load SST-2, tokenize, and return train/validation DataLoaders.

    Args:
        config: full config dict
        data_fraction: fraction of training data to use (for ablation)

    Returns:
        train_loader, val_loader
    """
    logger.info("Loading SST-2 dataset...")
    raw = load_dataset("glue", "sst2")

    tokenizer = get_tokenizer(config["model"]["pretrained"])
    max_length = config["data"]["max_length"]
    batch_size = config["training"]["batch_size"]

    train_data = raw["train"]
    if data_fraction < 1.0:
        n = int(len(train_data) * data_fraction)
        train_data = train_data.select(range(n))
        logger.info(f"Using {data_fraction*100:.0f}% of training data: {n} samples")

    train_tokenized = tokenize_dataset(train_data, tokenizer, max_length)
    val_tokenized   = tokenize_dataset(raw["validation"], tokenizer, max_length)

    train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_tokenized,   batch_size=batch_size, shuffle=False)

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader
