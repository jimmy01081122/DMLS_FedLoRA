# Federated LoRA Study: Final Implementation & Verification Report

This project evaluates LoRA variants in a robust Federated Learning (FL) framework optimized for resource-constrained hardware (NVIDIA RTX 3050 6GB).

## 1. Final Experimental Findings
The experiment successfully compared **Standard LoRA**, **FFA-LoRA**, and **RoLoRA** across three levels of data heterogeneity ($\alpha \in \{10.0, 0.5, 0.1\}$) under a simulated **20% Client Dropout rate**.

### 1.1 Performance Highlights (Alpha=10.0)
- **Baseline Accuracy**: Standard LoRA achieved **85.46%** test accuracy.
- **Communication Efficiency**: 
    - **FFA-LoRA**: **62.8% savings** (85.6 MB vs 230 MB).
    - **RoLoRA**: **52.3% savings** (109.7 MB vs 230 MB).
- **Aggregation Bias**: 
    - Standard LoRA: ~0.1 (Non-zero).
    - FFA/RoLoRA: **0.0000** (Zero Bias).
    - **Reasoning**: In FFA and RoLoRA, at least one LoRA matrix (A or B) is frozen during each aggregation round. This makes the product delta W = BA linear with respect to the updated parameters, allowing the average of products to equal the product of averages.

### 1.3 The Non-IID Challenge
- **Observation**: Accuracy drops significantly when $\alpha \le 0.5$.
- **Reasoning**: This is primarily due to **Client Drift** and **Classification Head conflict**. In extreme scenarios, local updates are mutually exclusive, leading to information washout during aggregation.
- **Recommendation**: Future audits should increase `GLOBAL_ROUNDS` to 50+ or implement drift-correction algorithms like **SCAFFOLD**.

## 2. Reproduction & Verification Guide
To verify the results and system robustness:

### 2.1 Visual Verification
- Open `results/accuracy_comparison.png` to view the performance matrix.
- Open `results/convergence_curves.png` to observe learning stability under dropout.
- Open `results/comm_breakdown.png` for communication cost analysis.

### 2.2 Log-based Verification
- **FedProx**: Search `logs/full_experiment.log` for local training logs. Verify the CPU-GPU parameter swapping logic in `src/main.py`.
- **Client Dropout**: Search for `[DROPOUT]` in the logs to see instances of simulated network failure.
- **Heterogeneous Aggregation**: Verify the shape matching logic in `fedavg_heterogeneous_states` within `src/main.py`.

### 2.3 Running the Environment
```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Execute (Quick mode for logic check, Full mode for data reproduction)
python3 src/main.py

# Plot
python3 src/plot_results.py
```

## 3. Project Deliverables
- `docs/report_en.tex`: 5+ page academic report (LaTeX).
- `docs/report_zh.md`: Comprehensive research report (Chinese).
- `results/`: Final data charts and CSVs.
- `checkpoints/`: Model states for all 5 rounds across all methods/alphas.

---
**Maintained by**: Distributed Machine Learning Laboratory (Gemini CLI Agent)
**Date**: May 17, 2026
