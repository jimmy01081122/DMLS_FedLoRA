#  LoRA 

## 
 RTX 3050 GPU LoRA Standard LoRAFFA-LoRA  RoLoRA Qwen2.5-1.5B  AG News  Non-IID FFA-LoRA  RoLoRA  Standard LoRA  Aggregation Bias 

## 1. 
LLMFederated LearningPEFT LoRA LoRA  Non-IID 

## 2. 
### 2.1 
- ****Qwen2.5-1.5B (4-bit NF4 )
- ****AG News ()
- **Non-IID ** Dirichlet  ($\alpha=0.1, 0.5, 1.0, 10.0$)  5 

### 2.2 
- **Standard LoRA** $A$  $B$ 
- **FFA-LoRA** $A$ $B$ 
- **RoLoRA** $A$  $B$

## 3. 
()

### 3.1 
|  |  |  (MB) |  |
| :--- | :--- | :--- | :--- |
| Standard LoRA | 0.3750 | 50.16 | 0% |
| FFA-LoRA | 0.0000 | 18.66 | 62.8% |
| RoLoRA | 0.0000 | 31.78 | 36.6% |

*10 /*

### 3.2 Non-IID 
 $\alpha=0.1$  Non-IID Standard LoRA  [METHOD] 

### 3.3 Aggregation Bias 
Standard LoRA  $A$  $B$ FedAvg  FFA-LoRA  RoLoRA 

## 4. 
1. ****FFA-LoRA  RoLoRA  LoRA  [DATA]%
2. ****RoLoRA  FFA-LoRA 
3. **** Non-IID [METHOD] 

## 5. 
 VRAM  RoLoRA  FFA-LoRA
