# BERT Fine-tuning Strategy Study

An empirical benchmark comparing five fine-tuning strategies for BERT on the SST-2 sentiment classification task, focusing on the trade-offs between parameter efficiency and model performance.

## Overview

This project systematically studies how different fine-tuning approaches affect accuracy, parameter efficiency, and data efficiency when adapting `bert-base-uncased` to downstream NLP tasks.

**Strategies compared:**

| Strategy | Trainable Layers | Method |
|---|---|---|
| `full_finetune` | All layers | Full fine-tuning (baseline) |
| `frozen_6_layers` | Last 6 encoder layers + classifier | Layer freezing |
| `frozen_10_layers` | Last 2 encoder layers + classifier | Layer freezing |
| `lora_r8` | LoRA adapters (r=8) + classifier | Low-Rank Adaptation |
| `lora_r16` | LoRA adapters (r=16) + classifier | Low-Rank Adaptation |

**Research questions:**
- What is the accuracy cost of freezing transformer layers?
- Can LoRA match full fine-tuning with far fewer trainable parameters?
- Which strategy is most sample-efficient under limited data?

## Project Structure

```
BERT-Finetuning-Strategy-Study/
├── configs/
│   ├── base.yaml               # Shared training hyperparameters
│   ├── full_finetune.yaml
│   ├── frozen_6_layers.yaml
│   ├── frozen_10_layers.yaml
│   ├── lora_r8.yaml
│   └── lora_r16.yaml
├── src/
│   ├── models.py               # Model builder (BERT + LoRA / frozen variants)
│   ├── lora.py                 # Custom LoRA implementation (no peft dependency)
│   ├── data.py                 # SST-2 data loading and tokenization
│   ├── trainer.py              # Training loop, AdamW + linear warmup
│   └── utils.py                # Config loading, logging, metrics, seeding
├── scripts/
│   ├── train.py                # Train a single strategy
│   ├── evaluate.py             # Compare all strategies, generate plots
│   └── ablation.py             # Data efficiency study (10% / 50% / 100%)
├── requirements.txt
└── setup.py
```

## Setup

**Requirements:** Python >= 3.8, PyTorch >= 2.0

```bash
git clone https://github.com/your-username/BERT-Finetuning-Strategy-Study.git
cd BERT-Finetuning-Strategy-Study
pip install -r requirements.txt
```

## Usage

### Train a single strategy

```bash
python scripts/train.py --config configs/full_finetune.yaml
python scripts/train.py --config configs/frozen_6_layers.yaml
python scripts/train.py --config configs/frozen_10_layers.yaml
python scripts/train.py --config configs/lora_r8.yaml
python scripts/train.py --config configs/lora_r16.yaml
```

Each run writes to `results/` (JSON + CSV metrics, log file).

### Compare all strategies

```bash
# Run after all five training jobs complete
python scripts/evaluate.py
```

Generates:
- `figures/accuracy_comparison.png` — bar chart of final validation accuracy
- `figures/params_vs_accuracy.png` — parameter efficiency scatter plot

### Data efficiency ablation

```bash
python scripts/ablation.py
```

Trains all five strategies at 10%, 50%, and 100% of the training set (3 epochs each) and produces `figures/data_efficiency.png`.

## Configuration

Training uses a two-level YAML config system. `configs/base.yaml` holds shared defaults:

```yaml
training:
  batch_size: 32
  num_epochs: 5
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  seed: 42
data:
  max_length: 128
```

Strategy configs override or extend these values. To add a new strategy, create a new YAML file and pass it via `--config`.

## LoRA Implementation

LoRA is implemented from scratch in [`src/lora.py`](src/lora.py) without the `peft` library. A `LoRALinear` wrapper replaces the query and value projection matrices in each attention layer:

```
output = W_original(x) + (x @ A @ B) * (alpha / r)
```

where `A` and `B` are low-rank matrices initialized as Gaussian and zero respectively.

## Outputs

| File | Description |
|---|---|
| `results/{strategy}.json` | Final metrics + per-epoch training history |
| `results/{strategy}_log.csv` | Epoch-level loss, accuracy, F1, runtime |
| `results/{strategy}.log` | Full training log |
| `figures/accuracy_comparison.png` | Strategy accuracy comparison |
| `figures/params_vs_accuracy.png` | Trainable params vs. accuracy |
| `figures/data_efficiency.png` | Accuracy vs. training data fraction |

## Dependencies

```
torch >= 2.0
transformers >= 4.35
datasets >= 2.14
scikit-learn >= 1.3
matplotlib >= 3.7
numpy >= 1.24
pyyaml >= 6.0
```

## Author

**Ada Yang** — adayang.chun@gmail.com
