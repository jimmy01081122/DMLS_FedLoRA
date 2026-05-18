import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import re

def parse_log_for_history(log_path):
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Split by Alpha
    alpha_blocks = re.split(r'Starting Alpha: ', content)[1:]
    
    history = {} # history[alpha][method] = list of metrics
    
    for block in alpha_blocks:
        lines = block.split('\n')
        alpha = lines[0].strip()
        history[alpha] = {}
        
        # Split by Method
        method_blocks = re.split(r'--- Method: ', block)[1:]
        for m_block in method_blocks:
            m_lines = m_block.split('\n')
            method = m_lines[0].strip()
            history[alpha][method] = []
            
            # Find Round lines
            for line in m_lines:
                if line.startswith("Round "):
                    # Round 5 - Acc: 0.8546, F1: 0.8538, Loss: 0.4099, Comm: 46.00MB, Bias: 0.037139
                    match = re.search(r'Round (\d+) - Acc: ([\d.]+), F1: ([\d.]+), Loss: ([\d.]+), Comm: ([\d.]+)MB, Bias: ([\d.]+)', line)
                    if match:
                        history[alpha][method].append({
                            "round": int(match.group(1)),
                            "acc": float(match.group(2)),
                            "f1": float(match.group(3)),
                            "loss": float(match.group(4)),
                            "comm_mb": float(match.group(5)),
                            "bias": float(match.group(6))
                        })
    return history

def plot_comprehensive_results(results_dir="results", log_path="logs/full_experiment.log"):
    # 1. Load the final summary results
    summary_path = os.path.join(results_dir, "final_results.csv")
    if not os.path.exists(summary_path):
        print("Final results summary not found.")
        return
    
    df = pd.read_csv(summary_path)
    history = parse_log_for_history(log_path)
    
    # Set professional style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 10, 'figure.titlesize': 12, 'axes.grid': True})
    
    # --------------------------------------------------
    # Plot 1: Accuracy across Methods and Alphas (Bar Chart)
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    methods = df['method'].unique()
    alphas = sorted(df['alpha'].unique(), reverse=True)
    
    n_groups = len(alphas)
    index = np.arange(n_groups)
    bar_width = 0.2
    
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method].sort_values('alpha', ascending=False)
        plt.bar(index + i*bar_width, method_data['final_acc'], bar_width, label=method.upper())
    
    plt.xlabel('Data Heterogeneity (Dirichlet Alpha)')
    plt.ylabel('Final Test Accuracy')
    plt.title('Performance comparison under varying Non-IID levels')
    plt.xticks(index + bar_width * (len(methods)-1)/2, [f'Alpha={a}' for a in alphas])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'), dpi=300)
    plt.close()

    # --------------------------------------------------
    # Plot 2: Learning Curves (Accuracy vs Rounds) from Log
    # --------------------------------------------------
    plt.figure(figsize=(12, 5))
    for idx, alpha_key in enumerate(history.keys()):
        plt.subplot(1, 3, idx + 1)
        for method, hist in history[alpha_key].items():
            if not hist: continue
            rounds = [h['round'] for h in hist]
            accs = [h['acc'] for h in hist]
            plt.plot(rounds, accs, marker='o', label=method)
            
        plt.title(f"Alpha={alpha_key}")
        plt.xlabel("Global Round")
        if idx == 0: plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Federated Learning Convergence with FedProx & Dropout")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'convergence_curves.png'), dpi=300)
    plt.close()

    # --------------------------------------------------
    # Plot 3: Communication Volume (Cumulative)
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    alpha_ref = list(history.keys())[0]
    for method, hist in history[alpha_ref].items():
        if not hist: continue
        # Cumulative communication for the last round
        total_comm = sum(h['comm_mb'] for h in hist)
        plt.bar(method.upper(), total_comm, label=method)
        
    plt.ylabel("Total Comm Volume (MB)")
    plt.title("Cumulative Communication Cost (5 Rounds)")
    plt.savefig(os.path.join(results_dir, 'comm_breakdown.png'), dpi=300)
    plt.close()

    # --------------------------------------------------
    # Plot 4: Aggregation Bias vs Global Communication Rounds
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    for method, hist in history[alpha_ref].items():
        if not hist: continue
        rounds = [h['round'] for h in hist]
        biases = [h['bias'] for h in hist]
        plt.plot(rounds, biases, marker='o', label=method.upper())
        
    plt.ylabel("Mean Aggregation Bias")
    plt.xlabel("Global Round")
    plt.title("Aggregation Bias vs. Global Communication Rounds (Alpha=10.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'bias_curves.png'), dpi=300)
    plt.close()

    # --------------------------------------------------
    # Plot 5: Label Distribution Matrix
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, alpha_val in enumerate(alphas):
        dist_path = os.path.join(results_dir, f"label_dist_alpha{alpha_val}.csv")
        if os.path.exists(dist_path):
            dist_df = pd.read_csv(dist_path, index_col=0)
            dist_df.plot(kind='bar', stacked=True, ax=axes[i], legend=(i==2))
            axes[i].set_title(f"Alpha = {alpha_val}")
            axes[i].set_xlabel("Client ID")
            if i == 0:
                axes[i].set_ylabel("Number of Samples")
            if i == 2:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle("Client Label Distribution Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'label_distribution.png'), dpi=300)
    plt.close()

    print("Comprehensive plots regenerated from logs/ directory.")

if __name__ == "__main__":
    plot_comprehensive_results()
