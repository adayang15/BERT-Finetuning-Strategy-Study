# BERT Finetuning Strategy Study

Empirical comparison of five fine-tuning strategies for BERT on the SST-2 binary sentiment classification task. The study benchmarks full fine-tuning against two layer-freezing variants and two LoRA configurations, examining the trade-offs between trainable parameter count, classification accuracy, and data efficiency.

## Topics Covered

**Dataset and Preprocessing**
The SST-2 dataset from the GLUE benchmark is loaded via HuggingFace Datasets and tokenized using the `bert-base-uncased` WordPiece tokenizer with padding and truncation to a maximum sequence length of 128 tokens. The data loader supports a fractional subset parameter to enable data efficiency experiments across different training set sizes.

**Fine-tuning Strategies**
Five strategies are implemented and compared:

- *Full Fine-tuning:* All parameters — embeddings, all twelve encoder layers, and the classification head — are updated during training. This serves as the performance upper bound.
- *Layer Freezing (6 layers):* Embeddings and the first six encoder layers are frozen. The upper six encoder layers and classification head remain trainable.
- *Layer Freezing (10 layers):* A more aggressive schedule that keeps only the last two encoder layers and classification head trainable.
- *LoRA rank 8 and rank 16:* Trainable low-rank adapter matrices are injected into the query and value projection layers of each attention head, leaving all original weights frozen. Adapters are initialized to contribute nothing at the start of training, preserving pretrained representations. Both ranks are compared to assess sensitivity to the size of the low-rank decomposition. The implementation is written from scratch without the `peft` library.

**Training Setup**
All strategies share a common configuration: AdamW optimizer, linear learning rate warmup over the first 10% of training steps followed by linear decay, batch size of 32, maximum gradient norm of 1.0, and 5 training epochs. A fixed random seed ensures reproducibility. Per-epoch accuracy, macro F1 score, loss, and wall-clock time are saved to `results/` as both JSON and CSV.

**Evaluation and Visualization**
After training, an evaluation script aggregates all five results and generates two figures: a validation accuracy bar chart and a trainable parameter count versus accuracy scatter plot on a log scale. The scatter plot makes the accuracy-efficiency frontier directly comparable across strategies.

**Data Efficiency Ablation**
All five strategies are retrained on 10%, 50%, and 100% of the SST-2 training set for 3 epochs each. The resulting learning curves reveal how quickly each strategy reaches competitive accuracy as labeled data increases, indicating which approaches are most suitable when labeled data is scarce.
