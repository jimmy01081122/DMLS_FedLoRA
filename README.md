# Federated LoRA: Robust Fine-tuning in Heterogeneous Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![VRAM 6GB](https://img.shields.io/badge/VRAM-6GB-green.svg)](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3050/)

This project investigates the performance, robustness, and efficiency of various LoRA-based Federated Fine-tuning methods under extreme resource constraints and data heterogeneity.

## Key Features
- **LoRA Variants**: Implements Standard LoRA, FFA-LoRA (Frozen-A), and RoLoRA (Rotating).
- **Robustness Suite**: 
    - **FedProx**: Memory-efficient L2 regularization for Non-IID data.
    - **Client Dropout**: Simulates 20% network failure rate.
    - **Dynamic Rank**: Supports heterogeneous clients ($r=4$ vs $r=8$) via Zero-padding aggregation.
- **Hardware Optimization**: Designed for RTX 3050 (6GB) using 4-bit NF4 quantization and PagedAdamW8bit.

## Quick Results (Alpha=10.0)
| Method | Accuracy | Comm Savings | Aggregation Bias |
| :--- | :--- | :--- | :--- |
| **Standard LoRA** | 85.46% | 0% | ~0.1 |
| **FFA-LoRA** | 81.78% | **62.8%** | **0.00** |
| **RoLoRA** | 80.91% | **52.3%** | **0.00** |

> **Note**: Accuracy significantly degrades in extreme Non-IID scenarios (alpha=0.1) due to inherent Client Drift, highlighting the need for advanced drift-correction in edge networks.

## Getting Started
### 1. Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Research
```bash
# Main experiment
python3 src/main.py

# Generate report-ready plots
python3 src/plot_results.py
```

## Project Structure
- `src/main.py`: Core FL framework and PEFT logic.
- `src/plot_results.py`: Visualization suite.
- `docs/`: Academic reports (LaTeX/Markdown).
- `results/`: CSVs and analytical charts.
- `checkpoints/`: Model states for reproducibility.

## Documentation
For a detailed mathematical framework and analysis, please refer to:
- [English Academic Report (LaTeX)](docs/report_en.tex)
- [Chinese Research Report (Markdown)](docs/report_zh.md)
- [Verification Guide (SUMMARY.md)](SUMMARY.md)

## License
This project is licensed under the MIT License.
