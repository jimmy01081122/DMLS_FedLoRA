import os
import gc
import copy
import time
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Transformers & PEFT
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score

# ==================================================
# CONFIGURATION
# ==================================================

class CONFIG:
    # Experiment Mode: "quick" or "full"
    MODE = "full" 
    
    # Model & Data
    MODEL_NAME = "Qwen/Qwen2.5-1.5B"
    DATASET_NAME = "ag_news"
    NUM_LABELS = 4
    MAX_SEQ_LENGTH = 128
    
    # Federated Learning
    NUM_CLIENTS = 5
    CLIENTS_PER_ROUND = 3
    GLOBAL_ROUNDS = 5
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 1
    LEARNING_RATE = 5e-5
    
    # Non-IID
    ALPHA_VALUES = [10.0, 0.5, 0.1]
    
    # Methods to compare
    METHODS = ["standard_lora", "ffa_lora", "rolora"]
    
    # Quantization
    LOAD_IN_4BIT = True
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_COMPUTE_DTYPE = torch.float16
    BNB_4BIT_USE_DOUBLE_QUANT = True
    
    # LoRA
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Aggregation Bias
    BIAS_MODE = "sampled_layers"
    MAX_BIAS_LAYERS = 4
    
    # Checkpointing
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    
    # Sample Limits
    MAX_TRAIN_SAMPLES_PER_CLIENT = 300
    MAX_EVAL_SAMPLES = 1000
    
    SEED = 42

if CONFIG.MODE == "quick":
    CONFIG.ALPHA_VALUES = [0.5]
    CONFIG.GLOBAL_ROUNDS = 1
    CONFIG.CLIENTS_PER_ROUND = 3
    CONFIG.MAX_TRAIN_SAMPLES_PER_CLIENT = 10
    CONFIG.MAX_EVAL_SAMPLES = 32 # Small multiple of batch size

# Create directories
os.makedirs(CONFIG.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CONFIG.RESULTS_DIR, exist_ok=True)

# Set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG.SEED)

# ==================================================
# PHASE 4 & 5: DATASET & PREPROCESSING
# ==================================================

def load_and_preprocess_data():
    """Load AG News and tokenize it."""
    print("Loading dataset...")
    dataset = load_dataset(CONFIG.DATASET_NAME)
    
    print(f"Loading tokenizer: {CONFIG.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=CONFIG.MAX_SEQ_LENGTH
        )
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    return tokenized_datasets, tokenizer

# ==================================================
# PHASE 6: DIRICHLET NON-IID SPLIT
# ==================================================

def dirichlet_split_noniid(labels, num_clients, alpha, seed=42):
    """Split data indices using Dirichlet distribution."""
    np.random.seed(seed)
    num_classes = len(np.unique(labels))
    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    client_indices = [[] for _ in range(num_clients)]
    
    for i in range(num_classes):
        # Dirichlet distribution for each class
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Allocation based on proportions
        proportions = (np.cumsum(proportions) * len(label_indices[i])).astype(int)[:-1]
        split_indices = np.split(label_indices[i], proportions)
        
        for j in range(num_clients):
            client_indices[j].extend(split_indices[j].tolist())
            
    for j in range(num_clients):
        np.random.shuffle(client_indices[j])
        
    return client_indices

def get_label_distribution(client_indices, labels):
    """Compute label distribution for each client."""
    distribution = []
    for i, indices in enumerate(client_indices):
        client_labels = labels[indices]
        counts = np.bincount(client_labels, minlength=CONFIG.NUM_LABELS)
        distribution.append(counts)
    return np.array(distribution)

# ==================================================
# PHASE 8 & 9: MODEL & PEFT SETUP
# ==================================================

def load_base_model():
    """Load the 4-bit quantized base model."""
    print(f"Loading model in 4-bit: {CONFIG.MODEL_NAME}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG.LOAD_IN_4BIT,
        bnb_4bit_quant_type=CONFIG.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=CONFIG.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=CONFIG.BNB_4BIT_USE_DOUBLE_QUANT,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG.MODEL_NAME,
        num_labels=CONFIG.NUM_LABELS,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.config.pad_token_id = model.config.eos_token_id
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model

def apply_lora(model):
    """Apply LoRA configuration to the model."""
    lora_config = LoraConfig(
        r=CONFIG.LORA_R,
        lora_alpha=CONFIG.LORA_ALPHA,
        target_modules=CONFIG.TARGET_MODULES,
        lora_dropout=CONFIG.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(model, lora_config)
    
    # [FIX 3] Initialize classification head properly to avoid insane initial loss
    for name, param in model.named_parameters():
        if is_classification_head(name):
            param.requires_grad = True
            torch.nn.init.normal_(param, mean=0.0, std=0.01)
            print(f"Initialized {name} with small weights.")
            
    return model

# ==================================================
# PHASE 10: UTILITY FUNCTIONS
# ==================================================

def is_classification_head(name):
    """Detect if a parameter is part of the classification head."""
    head_keywords = ["score", "classifier", "classification_head"]
    if any(kw in name for kw in head_keywords) and "lora_" not in name:
        return True
    return False

def get_trainable_params(model):
    """Return list of trainable parameter names."""
    return [name for name, param in model.named_parameters() if param.requires_grad]

def set_requires_grad(model, method, round_idx):
    """Control requires_grad based on method and round."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            if method == "standard_lora":
                param.requires_grad = True
            elif method == "ffa_lora":
                # Only train lora_B
                param.requires_grad = "lora_B" in name
            elif method == "rolora":
                # [FIX 4] Start with lora_B in round 1 to avoid zero gradient if B is all-zero
                if round_idx % 2 != 0:
                    param.requires_grad = "lora_B" in name
                else:
                    param.requires_grad = "lora_A" in name
        elif is_classification_head(name):
            param.requires_grad = True
        else:
            param.requires_grad = False

def should_transmit_param(name, method, round_idx):
    """Determine if a parameter should be transmitted."""
    if is_classification_head(name):
        return True
    if "lora_" in name:
        if method == "standard_lora":
            return True
        elif method == "ffa_lora":
            return "lora_B" in name
        elif method == "rolora":
            if round_idx % 2 != 0:
                return "lora_B" in name
            else:
                return "lora_A" in name
    return False

# ==================================================
# PHASE 11: WEIGHT MANAGEMENT
# ==================================================

def extract_transmitted_state(model, method, round_idx):
    """Extract parameters that need to be transmitted and move to CPU."""
    state_dict = {}
    for name, param in model.named_parameters():
        if should_transmit_param(name, method, round_idx):
            state_dict[name] = param.detach().cpu().clone()
    return state_dict

def load_global_state(model, global_state):
    """Load aggregated global state back into the model."""
    model_state = model.state_dict()
    for name, param in global_state.items():
        if name in model_state:
            model_state[name].copy_(param)
    model.load_state_dict(model_state, strict=False)

def fedavg_states(client_states, client_sizes):
    """Perform weighted FedAvg on the client states on CPU."""
    total_samples = sum(client_sizes)
    weights = [n / total_samples for n in client_sizes]
    
    global_state = {}
    # Use the keys from the first client
    keys = client_states[0].keys()
    
    for key in keys:
        global_state[key] = torch.zeros_like(client_states[0][key])
        for i in range(len(client_states)):
            global_state[key] += weights[i] * client_states[i][key]
            
    return global_state

# ==================================================
# PHASE 12: COMMUNICATION COST
# ==================================================

def calculate_communication_cost(state_dict):
    """Calculate communication cost in MB."""
    total_bytes = 0
    lora_a_bytes = 0
    lora_b_bytes = 0
    head_bytes = 0
    
    for name, param in state_dict.items():
        element_size = 2 if param.dtype == torch.float16 or param.dtype == torch.bfloat16 else 4
        param_bytes = param.numel() * element_size
        total_bytes += param_bytes
        
        if "lora_A" in name:
            lora_a_bytes += param_bytes
        elif "lora_B" in name:
            lora_b_bytes += param_bytes
        elif is_classification_head(name):
            head_bytes += param_bytes
            
    return {
        "total_mb": total_bytes / (1024 * 1024),
        "lora_a_mb": lora_a_bytes / (1024 * 1024),
        "lora_b_mb": lora_b_bytes / (1024 * 1024),
        "head_mb": head_bytes / (1024 * 1024),
        "transmitted_params": sum(p.numel() for p in state_dict.values())
    }

# ==================================================
# PHASE 13: AGGREGATION BIAS
# ==================================================

def calculate_aggregation_bias(client_states, global_state, method, round_idx):
    """Calculate Aggregation Bias layer-by-layer to save memory."""
    bias_values = []
    
    lora_layers = []
    sample_state = client_states[0]
    for name in sample_state.keys():
        if "lora_A" in name:
            layer_base = name.replace("lora_A.default.weight", "")
            if layer_base not in lora_layers:
                lora_layers.append(layer_base)
            
    if CONFIG.BIAS_MODE == "sampled_layers":
        random.seed(CONFIG.SEED)
        if len(lora_layers) > CONFIG.MAX_BIAS_LAYERS:
            lora_layers = sorted(lora_layers)
            indices = np.linspace(0, len(lora_layers)-1, CONFIG.MAX_BIAS_LAYERS).astype(int)
            lora_layers = [lora_layers[i] for i in indices]

    for layer in lora_layers:
        a_key = layer + "lora_A.default.weight"
        b_key = layer + "lora_B.default.weight"
        
        if a_key not in client_states[0] or b_key not in client_states[0]:
            continue
            
        num_clients = len(client_states)
        delta_w_sum = None
        
        for k in range(num_clients):
            A_k = client_states[k][a_key].to(torch.float32)
            B_k = client_states[k][b_key].to(torch.float32)
            W_k = B_k @ A_k
            
            if delta_w_sum is None:
                delta_w_sum = W_k
            else:
                delta_w_sum += W_k
            
            del A_k, B_k, W_k
        
        delta_w_ideal = delta_w_sum / num_clients
        
        if a_key in global_state and b_key in global_state:
            A_avg = global_state[a_key].to(torch.float32)
            B_avg = global_state[b_key].to(torch.float32)
            delta_w_fedavg = B_avg @ A_avg
            
            bias = torch.norm(delta_w_ideal - delta_w_fedavg) / (torch.norm(delta_w_ideal) + 1e-8)
            bias_values.append(bias.item())
            
            del A_avg, B_avg, delta_w_fedavg
        
        del delta_w_sum, delta_w_ideal
        gc.collect()
        
    return np.mean(bias_values) if bias_values else 0.0

# ==================================================
# PHASE 14 & 15: TRAINING & EVALUATION
# ==================================================

def local_train(model, train_dataset, method, round_idx):
    """Train the model locally for a client."""
    set_requires_grad(model, method, round_idx)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG.LEARNING_RATE)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG.BATCH_SIZE, 
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer)
    )
    
    model.train()
    total_loss = 0
    
    optimizer.zero_grad()
    actual_steps = 0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / CONFIG.GRAD_ACCUM_STEPS
        
        # [DEBUG] Check for NaN loss
        if torch.isnan(loss):
            print("  [WARNING] NaN loss detected in local train!")
            
        loss.backward()
        
        if (step + 1) % CONFIG.GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            actual_steps += 1
            
        total_loss += loss.item() * CONFIG.GRAD_ACCUM_STEPS
        
        if CONFIG.MODE == "quick" and step >= CONFIG.MAX_TRAIN_SAMPLES_PER_CLIENT:
            break
            
    # Cleanup
    del optimizer, train_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_loss / (step + 1)

def evaluate_model(model, eval_dataset, tokenizer):
    """Evaluate the model on the evaluation dataset."""
    model.eval()
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=CONFIG.BATCH_SIZE * 4, 
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer)
    )
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_loader, desc="Evaluating", leave=False)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            total_loss += outputs.loss.item()
            
            if CONFIG.MODE == "quick" and step >= CONFIG.MAX_EVAL_SAMPLES // (CONFIG.BATCH_SIZE * 4):
                break
                
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / (step + 1)
    
    return acc, f1, avg_loss

# ==================================================
# PHASE 16: CHECKPOINTING
# ==================================================

def save_checkpoint(method, alpha, round_idx, global_state, metrics, comm_logs, bias_logs):
    """Save experiment checkpoint."""
    ckpt_path = os.path.join(CONFIG.CHECKPOINT_DIR, f"{method}_alpha{alpha}_round{round_idx}.pt")
    checkpoint = {
        "method": method,
        "alpha": alpha,
        "round_idx": round_idx,
        "global_state": global_state,
        "metrics": metrics,
        "comm_logs": comm_logs,
        "bias_logs": bias_logs,
        "config": {k: v for k, v in CONFIG.__dict__.items() if not k.startswith("__")},
        "seed": CONFIG.SEED
    }
    torch.save(checkpoint, ckpt_path)

def load_checkpoint(path):
    return torch.load(path)

# ==================================================
# PHASE 17 & 18: EXPERIMENT LOOP
# ==================================================

def run_experiment():
    tokenized_datasets, tokenizer = load_and_preprocess_data()
    train_labels = np.array(tokenized_datasets["train"]["labels"])
    
    results_all = []
    
    for alpha in CONFIG.ALPHA_VALUES:
        print(f"\n{'='*50}\nStarting Alpha: {alpha}\n{'='*50}")
        client_indices = dirichlet_split_noniid(train_labels, CONFIG.NUM_CLIENTS, alpha, seed=CONFIG.SEED)
        
        # Log label distribution
        dist = get_label_distribution(client_indices, train_labels)
        dist_df = pd.DataFrame(dist, columns=["World", "Sports", "Business", "Sci/Tech"])
        dist_df.to_csv(os.path.join(CONFIG.RESULTS_DIR, f"label_dist_alpha{alpha}.csv"))
        print("Label Distribution:\n", dist_df)
        
        for method in CONFIG.METHODS:
            print(f"\n--- Method: {method} ---")
            
            # Reset model for each method/alpha
            base_model = load_base_model()
            model = apply_lora(base_model)
            
            metrics_history = []
            comm_history = []
            bias_history = []
            cumulative_comm = 0
            
            # Initial evaluation
            acc, f1, loss = evaluate_model(model, tokenized_datasets["test"], tokenizer)
            print(f"Initial - Acc: {acc:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}")
            
            for r in range(1, CONFIG.GLOBAL_ROUNDS + 1):
                print(f"Global Round {r}/{CONFIG.GLOBAL_ROUNDS}")
                
                # Save current global state BEFORE client loop
                current_global_state = extract_transmitted_state(model, "standard_lora", r)
                
                # Select clients
                selected_clients = random.sample(range(CONFIG.NUM_CLIENTS), CONFIG.CLIENTS_PER_ROUND)
                client_states = []
                client_sizes = []
                
                round_comm_mb = 0
                
                for client_id in selected_clients:
                    print(f"  Training Client {client_id}...")
                    
                    # Reload global state before each client
                    load_global_state(model, current_global_state)
                    
                    # Create client dataset
                    indices = client_indices[client_id]
                    indices = indices[:CONFIG.MAX_TRAIN_SAMPLES_PER_CLIENT]
                    client_dataset = Subset(tokenized_datasets["train"], indices)
                    
                    # Local Train
                    client_loss = local_train(model, client_dataset, method, r)
                    if r == 1:
                        print(f"    Client Loss: {client_loss:.4f}")
                    
                    # Extract state
                    state = extract_transmitted_state(model, method, r)
                    client_states.append(state)
                    client_sizes.append(len(indices))
                    
                    # Calculate comm for this client (upload)
                    cost = calculate_communication_cost(state)
                    round_comm_mb += cost["total_mb"]

                # FedAvg
                global_state = fedavg_states(client_states, client_sizes)
                
                # Calculate bias (before updating model with new global state)
                bias = calculate_aggregation_bias(client_states, global_state, method, r)
                bias_history.append({"round": r, "bias": bias})
                
                # Update global model
                load_global_state(model, global_state)
                
                # Download cost (server to all clients)
                cost_down = calculate_communication_cost(global_state)
                round_comm_mb += cost_down["total_mb"] * len(selected_clients)
                
                cumulative_comm += round_comm_mb
                comm_history.append({
                    "round": r, 
                    "round_mb": round_comm_mb, 
                    "cumulative_mb": cumulative_comm,
                    "lora_a_mb": cost_down["lora_a_mb"],
                    "lora_b_mb": cost_down["lora_b_mb"],
                    "head_mb": cost_down["head_mb"]
                })
                
                # Global Evaluation
                acc, f1, loss = evaluate_model(model, tokenized_datasets["test"], tokenizer)
                print(f"Round {r} - Acc: {acc:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}, Comm: {round_comm_mb:.2f}MB, Bias: {bias:.6f}")
                
                metrics_history.append({
                    "round": r, "acc": acc, "f1": f1, "loss": loss
                })
                
                # Save Checkpoint
                save_checkpoint(method, alpha, r, global_state, metrics_history, comm_history, bias_history)
                
            # Final Results for this method
            res = {
                "method": method,
                "alpha": alpha,
                "final_acc": acc,
                "final_f1": f1,
                "final_loss": loss,
                "best_acc": max(m["acc"] for m in metrics_history),
                "best_f1": max(m["f1"] for m in metrics_history),
                "total_comm_mb": cumulative_comm,
                "mean_bias": np.mean([b["bias"] for b in bias_history])
            }
            results_all.append(res)
            
            # Cleanup for next method
            del model, base_model
            gc.collect()
            torch.cuda.empty_cache()
            
    # Save all results
    results_df = pd.DataFrame(results_all)
    results_df.to_csv(os.path.join(CONFIG.RESULTS_DIR, "final_results.csv"))
    print("\nFinal Results Summary:\n", results_df)
    
    return results_df

# ==================================================
# PHASE 20: PLOTTING
# ==================================================

def plot_results(results_dir):
    pass

if __name__ == "__main__":
    run_experiment()
