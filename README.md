# BERT-Finetuning-Strategy-Study

Empirical comparison of five fine-tuning strategies for BERT on the SST-2 binary sentiment classification task. The study benchmarks full fine-tuning against two layer-freezing variants and two Low-Rank Adaptation (LoRA) configurations, examining the trade-offs between trainable parameter count, classification accuracy, and data efficiency across different training set sizes.

## Topics Covered

**Dataset and Preprocessing**
The SST-2 dataset from the GLUE benchmark is loaded via HuggingFace Datasets. Each sentence is tokenized using the `bert-base-uncased` WordPiece tokenizer with padding and truncation applied to a maximum sequence length of 128 tokens. To support data efficiency experiments, the data loader accepts a fractional subset parameter that samples a fixed proportion of the training split while preserving label distribution.

**Fine-tuning Strategies**

*Full Fine-tuning:* All parameters of `BertForSequenceClassification` — embeddings, all twelve encoder layers, and the classification head — are updated during training. This serves as the performance upper bound against which parameter-efficient methods are evaluated.

*Layer Freezing (6 layers):* The embedding layer and the first six transformer encoder layers are frozen. Only the upper six encoder layers and the classification head remain trainable. This reduces the number of gradient updates per step while retaining the higher-level contextual representations that are most task-relevant.

*Layer Freezing (10 layers):* A more aggressive freezing schedule that retains only the last two encoder layers and the classification head as trainable. This configuration tests how much task-specific adaptation can be achieved with a minimal slice of the network.

*LoRA (rank 8) and LoRA (rank 16):* Low-Rank Adaptation injects trainable low-rank matrices into the query and value projection layers of each attention head, leaving all original weights frozen. The forward pass computes `W_original(x) + (x @ A @ B) * (alpha / r)`, where `A` is initialized from a Gaussian distribution and `B` is initialized to zero, ensuring the adapter contributes nothing at the start of training. Rank 8 and rank 16 are compared to assess the sensitivity of LoRA to the expressiveness of its low-rank decomposition. The LoRA implementation is written from scratch without relying on the `peft` library.

**Training Setup**
All strategies share a common training configuration: AdamW optimizer, linear learning rate warmup over the first 10% of training steps followed by linear decay, batch size of 32, maximum gradient norm of 1.0, and a fixed random seed for reproducibility. Each strategy is trained for 5 epochs. Per-epoch accuracy, macro F1 score, loss, and wall-clock time are recorded to `results/` as both JSON and CSV.

**Evaluation and Visualization**
After all five strategies are trained, a separate evaluation script aggregates results and generates two figures: an accuracy comparison bar chart and a parameter efficiency scatter plot that places each strategy in the space of trainable parameter count versus validation accuracy on a log scale. This plot makes the accuracy-efficiency frontier directly visible.

**Data Efficiency Ablation**
An ablation study trains all five strategies on 10%, 50%, and 100% of the SST-2 training set for 3 epochs each, producing learning curves that show how quickly each strategy reaches competitive accuracy as labeled data increases. This reveals which approaches are most suitable when labeled data is scarce.
