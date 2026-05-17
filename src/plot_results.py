import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import torch

def plot_comprehensive_results(results_dir="results", checkpoint_dir="checkpoints"):
    # 1. Load the final summary results
    summary_path = os.path.join(results_dir, "final_results.csv")
    if not os.path.exists(summary_path):
        print("Final results summary not found. Attempting to aggregate from checkpoints...")
        # (Wait for experiment to finish or aggregate partial results if needed)
        return
    
    df = pd.read_csv(summary_path)
    
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
    # Plot 2: Learning Curves (Accuracy vs Rounds) from Checkpoints
    # --------------------------------------------------
    # This requires reading metrics_history from the latest checkpoint of each method
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        plt.subplot(1, len(alphas), alphas.index(alpha) + 1)
        for method in methods:
            # Find the latest checkpoint for this method and alpha
            ckpt_files = glob.glob(os.path.join(checkpoint_dir, f"{method}_alpha{alpha}_round*.pt"))
            if not ckpt_files: continue
            
            # Sort by round
            ckpt_files.sort(key=lambda x: int(x.split('round')[-1].split('.pt')[0]))
            latest_ckpt = torch.load(ckpt_files[-1], map_location='cpu')
            history = latest_ckpt['metrics']
            
            rounds = [h['round'] for h in history]
            accs = [h['acc'] for h in history]
            plt.plot(rounds, accs, marker='o', label=f"{method}")
            
        plt.title(f"Alpha={alpha}")
        plt.xlabel("Global Round")
        if alphas.index(alpha) == 0: plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Federated Learning Convergence with FedProx & Dropout")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'convergence_curves.png'), dpi=300)
    plt.close()

    # --------------------------------------------------
    # Plot 3: Communication Breakdown
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    for i, method in enumerate(methods):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, f"{method}_alpha{alphas[0]}_round*.pt"))
        if not ckpt_files: continue
        latest_ckpt = torch.load(ckpt_files[-1], map_location='cpu')
        comm = latest_ckpt['comm_logs'][-1]
        
        categories = ['LoRA A', 'LoRA B', 'Head']
        values = [comm['lora_a_mb'], comm['lora_b_mb'], comm['head_mb']]
        
        plt.bar(method.upper(), comm['cumulative_mb'], label=method)
        
    plt.ylabel("Total Comm Volume (MB)")
    plt.title("Cumulative Communication Cost Breakdown")
    plt.savefig(os.path.join(results_dir, 'comm_breakdown.png'), dpi=300)
    plt.close()

    print("Comprehensive plots generated in results/ directory.")

if __name__ == "__main__":
    plot_comprehensive_results()
