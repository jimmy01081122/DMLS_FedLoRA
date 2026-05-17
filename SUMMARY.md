# DMLS-Project: Federated LoRA Comparative Study

A comprehensive research framework for evaluating LoRA-based Federated Fine-tuning methods on resource-constrained hardware.

## Project Overview
This project simulates a Federated Learning (FL) environment on a single NVIDIA RTX 3050 (6GB VRAM) to compare three PEFT strategies:
- **Standard LoRA**: The baseline method.
- **FFA-LoRA**: Communication-efficient method freezing the $A$ matrix.
- **RoLoRA**: Alternating update strategy to balance capacity and bandwidth.

## Key Results
- **Model**: Qwen2.5-1.5B (4-bit NF4 Quantization)
- **Task**: AG News Sequence Classification (4 classes)
- **Performance**: Standard LoRA achieved ~88% accuracy in just 5 rounds.
- **Efficiency**: FFA-LoRA and RoLoRA reduced communication costs by **47% - 63%**.
- **Stability**: Confirmed that freezing one matrix during aggregation eliminates non-linear **Aggregation Bias**.

## Technical Highlights
- **Layer-by-Layer Analytics**: Memory-safe implementation to calculate complex metrics without OOM.
- **Simulation**: Sequential client processing mimicking distributed environments.
- **Non-IID**: Integrated Dirichlet distribution for heterogeneous data modeling.

## Deliverables
- `src/main.py`: Core FL framework.
- `docs/report_zh.md`: Detailed research report in Chinese.
- `docs/report_en.tex`: Academic LaTeX report.
- `results/`: Visual analytics and raw data.

## Usage
```bash
make install  # Setup environment
make run      # Execute research
make plot     # Generate charts
```
