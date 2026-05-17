# Federated LoRA Study: Implementation & Verification Guide

This document serves as a guide for reproducing the experimental results and verifying the system's robustness enhancements.

## 1. System Requirements & Setup
- **Hardware**: NVIDIA GPU with at least 6GB VRAM (e.g., RTX 3050).
- **RAM**: Minimum 16GB recommended.
- **Disk**: ~10GB for model weights and datasets.
- **Environment**: 
    - Python 3.10+
    - CUDA 11.8+
    - Dependencies listed in `requirements.txt`.

### 1.1 Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Running Experiments
The core logic is contained in `src/main.py`. You can toggle between `quick` and `full` modes in the `CONFIG` class.

### 2.1 Configuration Parameters
- `MU`: Controls the proximal term in FedProx (default: 0.01).
- `CLIENT_DROPOUT_RATE`: Probability of a client failing to upload updates (default: 0.2).
- `HETEROGENEOUS_RANKS`: List of ranks assigned to clients (e.g., `[4, 4, 8, 8, 8]`).

### 2.2 Execution
```bash
# Run the main experiment
python3 src/main.py
```

## 3. Verification of Robustness Features
To verify the individual components:
- **FedProx**: Check `local_train_fedprox` in `src/main.py`. It uses a memory-safe iteration over parameters.
- **Client Dropout**: Look for the `[DROPOUT]` logs during execution.
- **Heterogeneous Aggregation**: Verify `fedavg_heterogeneous_states` and the zero-padding logic.

## 4. Result Analysis
Final results are stored in `results/final_results.csv`. 
Run the following to generate plots:
```bash
python3 src/plot_results.py
```

## 5. Directory Structure
- `src/`: Core implementation.
- `docs/`: Academic reports and documentation.
- `results/`: Output charts and data.
- `checkpoints/`: Model states for each round.
- `logs/`: Execution logs.
