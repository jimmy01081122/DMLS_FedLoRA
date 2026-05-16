# LoRA-based Federated Fine-tuning Comparative Study

This project explores and compares three LoRA (Low-Rank Adaptation) variants in a Federated Learning (FL) context, specifically optimized for resource-constrained environments like a single NVIDIA RTX 3050 GPU (6GB VRAM).

## 🚀 Overview

The goal is to evaluate the performance, communication cost, and aggregation stability of different PEFT (Parameter-Efficient Fine-Tuning) methods under Non-IID data distributions using the **Qwen2.5-1.5B** model and the **AG News** dataset.

### Compared Methods:
1.  **Standard LoRA**: Full updates and transmission of $A$ and $B$ matrices.
2.  **FFA-LoRA (Frozen-A)**: Freezes matrix $A$ after initialization; only $B$ is trained and transmitted.
3.  **RoLoRA (Rotating LoRA)**: Alternates training between $A$ and $B$ across rounds to reduce per-round communication.

## 🛠️ Key Features

*   **4-bit Quantization**: Uses `bitsandbytes` (NF4) to fit large models in 6GB VRAM.
*   **Sequential Client Simulation**: Mimics a multi-client FL environment on a single GPU.
*   **Non-IID Simulation**: Implements Dirichlet distribution split ($\alpha \in \{10.0, 0.5, 0.1\}$).
*   **Memory-Efficient Analytics**: Layer-by-layer **Aggregation Bias** calculation to prevent OOM errors.
*   **Automated Reporting**: Generates both academic LaTeX (English) and general Markdown (Chinese) reports.

## 📁 Project Structure

```text
.
├── docs/               # Research reports and original prompt
├── src/                # Core Python scripts (FL loop, plotting)
├── results/            # Training logs, CSVs, and generated charts
├── checkpoints/        # Model adapter weights per round
├── Makefile            # One-click automation commands
└── requirements.txt    # Dependency list
```

## ⚡ Quick Start

### 1. Installation
Ensure you have Python 3.10+ and CUDA 12.x installed.
```bash
make install
```

### 2. Run Experiment
The default configuration is set to `full` research mode.
```bash
make run
```

### 3. Generate Charts
After the experiment completes, visualize the results:
```bash
make plot
```

## 📊 Hardware Constraints & Optimizations

This project is specifically tuned for the **NVIDIA RTX 3050 (6GB)**:
*   **Memory**: NF4 quantization + Gradient Checkpointing + Sequential Processing.
*   **Speed**: Optimized batch sizes and gradient accumulation steps.

## 📝 License
MIT License
