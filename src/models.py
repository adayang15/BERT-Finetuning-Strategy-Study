"""Model builder: returns the correct BERT variant based on strategy config."""

import logging
from transformers import BertForSequenceClassification
from src.lora import apply_lora

logger = logging.getLogger(__name__)


def build_model(config: dict):
    """
    Build and return a BertForSequenceClassification model based on strategy type.

    Strategy types:
        full    - all parameters trainable
        frozen  - freeze embeddings + first N encoder layers
        lora    - freeze all except LoRA adapters and classifier
    """
    pretrained  = config["model"]["pretrained"]
    num_labels  = config["model"]["num_labels"]
    strategy    = config["strategy"]
    stype       = strategy["type"]

    logger.info(f"Building model | strategy={stype} | pretrained={pretrained}")

    model = BertForSequenceClassification.from_pretrained(pretrained, num_labels=num_labels)

    if stype == "full":
        # All parameters remain trainable
        pass

    elif stype == "frozen":
        freeze_embeddings    = strategy.get("freeze_embeddings", True)
        freeze_encoder_layers = strategy.get("freeze_encoder_layers", 6)

        if freeze_embeddings:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        for i in range(freeze_encoder_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        logger.info(f"Frozen: embeddings={freeze_embeddings}, encoder layers 0-{freeze_encoder_layers-1}")

    elif stype == "lora":
        r              = strategy["lora_r"]
        alpha          = strategy["lora_alpha"]
        target_modules = strategy.get("lora_target", ["query", "value"])
        model = apply_lora(model, r=r, alpha=alpha, target_modules=target_modules)

    else:
        raise ValueError(f"Unknown strategy type: {stype}")

    return model
