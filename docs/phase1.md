PEFTLoRA RTX 3050 
 LoRA-based Federated Fine-tuning  Non-IID 

==================================================

==================================================



-  NVIDIA 3050
- VRAM
-  15GB  RAM
-  session timeout
-  client model
-  sequential client simulation
-  GPU / CPU 
-  checkpointing



1.  bitsandbytes 4-bit NF4 
2.  full fine-tune base model
3.  GPU  client model
4. client  CPU RAM 
5. FedAvg  CPU 
6.  client local training 
   - del dataloader  batch cache
   - del optimizer
   - del loss
   - torch.cuda.empty_cache()
   - gc.collect()

==================================================

==================================================



- DatasetAG News
- Task4-class text classification
- Model architectureSequence Classification
- Base modelQwen2.5-1.5B  1.5B  decoder-only model
- AutoModelForSequenceClassification
- num_labels = 4
- Quantization4-bit NF4
- PEFTLoRA



 Sequence Classification Next-Token Prediction / Causal LM label generation

 AG News Sequence Classification  4  logits accuracymacro-F1validation lossFedAvg 



4-bit quantized base LLM backbone
+ LoRA adapters
+ trainable classification head

 classification head  LoRA 

1. requires_grad=True
2.  client  local training
3.  server-side FedAvg
4.  communication cost
5.  checkpoint saving / loading

 LoRA 

==================================================

==================================================



--------------------------------------------------
1. Standard LoRA
--------------------------------------------------



- LoRA A 
- LoRA B 
- classification head 

 client 

- lora_A
- lora_B
- classification head

Server  weighted FedAvg 

--------------------------------------------------
2. FFA-LoRA
--------------------------------------------------



-  LoRA A
-  LoRA B
- classification head 

 client 

- lora_B
- classification head

Server  weighted FedAvg 



FFA-LoRA  LoRA adapter   
 classification head  Standard LoRA  50%



FFA total communication ratio =
(size(lora_B) + size(classification_head))
/
(size(lora_A) + size(lora_B) + size(classification_head))

--------------------------------------------------
3. RoLoRA
--------------------------------------------------



- 
-  global round B A
-  global round A B
- classification head 

 round client 

- lora_A
- classification head

 round client 

- lora_B
- classification head

Server  weighted FedAvg 



RoLoRA  LoRA adapter   
 classification head Standard LoRA  50%

==================================================

==================================================



==================================================
 vs 
==================================================



Standard LoRAFFA-LoRA  RoLoRA macro-F1validation loss 



1.  global round  validation / test set 
   - Accuracy
   - Macro-F1
   - Validation loss

2.  round 
   -  MB
   -  MB
   - trainable parameters
   - transmitted parameters
   - LoRA-only communication MB
   - total communication MB including classification head

3. 
   - Accuracy vs Global Round
   - Macro-F1 vs Global Round
   - Validation Loss vs Global Round
   - Cumulative Communication MB vs Accuracy
   - Cumulative Communication MB vs Macro-F1
   - Per-round Communication MB by Method

4.  summary table
   - Method
   - Final Accuracy
   - Final Macro-F1
   - Best Accuracy
   - Best Macro-F1
   - Final Validation Loss
   - Total Communication MB
   - LoRA-only Communication MB
   - Classification Head Communication MB
   - Trainable Parameters
   - Transmitted Parameters
   - Communication Saving Ratio compared with Standard LoRA



- FFA-LoRA  LoRA adapter  Standard LoRA  50%
- RoLoRA  LoRA adapter  Standard LoRA  50%
-  classification head  50%
-  accuracy  macro-F1 

==================================================
Non-IID 
==================================================





 Dirichlet distribution  5  virtual clients  Non-IID 

 Dirichlet alpha

- alpha = 10.0 IID
- alpha = 1.0 Non-IID
- alpha = 0.5 Non-IID
- alpha = 0.1 Non-IID



1. Dirichlet Non-IID split function
   - inputdataset labels, num_clients, alpha, seed
   - outputclient_indices
   -  client  min_samples
   -  client 

2.  alpha  client label distribution table
   - rowclient
   - columnsWorld, Sports, Business, Sci/Tech
   - values
   -  client  label proportion

3.  alpha 

4.  alpha 
   - Final Accuracy
   - Final Macro-F1
   - Best Accuracy
   - Best Macro-F1
   - Final Validation Loss
   - Accuracy standard deviation across rounds
   - Macro-F1 standard deviation across rounds
   - Total Communication MB
   - Mean Aggregation Bias

5. 
   - Alpha vs Final Accuracy
   - Alpha vs Final Macro-F1
   - Alpha vs Total Communication MB
   - Alpha vs Mean Aggregation Bias
   - Accuracy curves under different alpha values
   - Macro-F1 curves under different alpha values



- Standard LoRA  alpha 
- FFA-LoRA  Non-IID 
- RoLoRA  Non-IID  FFA-LoRA 
-  alpha = 0.1  alpha = 0.5  robustness

==================================================
LoRA  FedAvg  Aggregation Bias
==================================================



Standard LoRA  A  B  aggregation bias  
FFA-LoRA  RoLoRA 

 LoRA layerLoRA update 

Delta W_k = B_k A_k

 k  client

Standard LoRA  FedAvg 

A_avg = average(A_k)
B_avg = average(B_k)
Delta W_fedavg = B_avg A_avg

 client update 

Delta W_ideal = average(B_k A_k)

Aggregation Bias 

Aggregation Bias =
|| Delta W_ideal - Delta W_fedavg ||_F
/
|| Delta W_ideal ||_F



 materialize  layers  Delta W  
 Colab RAM  GPU VRAM OOM

 layer-by-layer 

 memory-safe aggregation bias function

1.  LoRA layer
2.  client state  layer  A_k  B_k
3.  CPU  GPU  B_k @ A_k
4.  aggregation bias scalar
5.  scalar  list
6.  del  tensor
7. 
   - gc.collect()
   - torch.cuda.empty_cache()
8. 



--------------------------------------------------
bias_mode = "sampled_layers"
--------------------------------------------------

 sampled_layers



-  2  LoRA layers
-  2  LoRA layers
-  2  LoRA layers

 sample 6  8  LoRA layers

--------------------------------------------------
bias_mode = "all_layers"
--------------------------------------------------





1. Standard LoRA aggregation bias
2. FFA-LoRA aggregation bias
3. RoLoRA aggregation bias

 FFA-LoRA

 A  A_0

average(B_k A_0)  average(B_k) A_0

 aggregation bias  0  Standard LoRA

 RoLoRA

 round  A B   
 round  B A 

 aggregation bias  Standard LoRA



- Aggregation Bias vs Global Round
- Aggregation Bias vs Accuracy
- Aggregation Bias vs Macro-F1
- Aggregation Bias under different Dirichlet alpha values

 summary table

- Method
- Alpha
- Mean Aggregation Bias
- Final Aggregation Bias
- Final Accuracy
- Final Macro-F1
- Total Communication MB



- Standard LoRA  aggregation bias  Non-IID 
- Aggregation bias  accuracy drop  training instability 
- FFA-LoRA  A  bias
- RoLoRA  aggregation bias 

==================================================
Optimizer 
==================================================

 RoLoRA  optimizer state 

 AdamWpaged_adamw_8bit  optimizer  momentum / variance RoLoRA  A/B  optimizer state optimizer state  global weights 



 client local training  optimizer



-  client  optimizer
-  global round  optimizer
-  client  optimizer
-  round  optimizer
- client  del optimizer

local_train function 

1. set_trainable_params(model, method, round_idx)
2. trainable_params = [p for p in model.parameters() if p.requires_grad]
3. optimizer = create_optimizer(trainable_params)
4. local training
5. extract transmitted state
6. del optimizer
7. gc.collect()
8. torch.cuda.empty_cache()

==================================================
Classification Head 
==================================================

 AutoModelForSequenceClassification  robust classification head detection function



is_classification_head(name)



- score.weight
- score.bias
- classifier.weight
- classifier.bias
- classification_head.*
- 

 LoRA  classification head



-  name  "lora_" classification head
-  name  "score""classifier""classification_head" classification head
-  trainable / transmitted parameter names

classification head 

1. requires_grad=True
2.  local training
3.  FedAvg
4.  communication cost
5.  checkpoint
6.  resume checkpoint 

==================================================
 FedAvg 
==================================================



--------------------------------------------------
1. should_train_param(name, method, round_idx)
--------------------------------------------------

 local training  requires_grad=True



Standard LoRA
- train lora_A
- train lora_B
- train classification head

FFA-LoRA
- freeze lora_A
- train lora_B
- train classification head

RoLoRA
- odd roundtrain lora_A + classification head
- even roundtrain lora_B + classification head

--------------------------------------------------
2. should_transmit_param(name, method, round_idx)
--------------------------------------------------

 client  server



Standard LoRA
- transmit lora_A
- transmit lora_B
- transmit classification head

FFA-LoRA
- transmit lora_B
- transmit classification head

RoLoRA
- odd roundtransmit lora_A + classification head
- even roundtransmit lora_B + classification head

--------------------------------------------------
3. extract_transmitted_state(model, method, round_idx)
--------------------------------------------------

 detachclone  CPU

 base model weights

--------------------------------------------------
4. load_global_state(model, global_state)
--------------------------------------------------

 server  global_state 

 LoRA  classification head

--------------------------------------------------
5. fedavg_states(client_states, client_sizes)
--------------------------------------------------

 CPU  weighted FedAvg



global_weight =
sum_k (n_k / total_n) * client_weight_k

 n_k  client k 

==================================================

==================================================

 communication cost function

 parameter  tensor element  dtype size 



1. LoRA A communication MB
2. LoRA B communication MB
3. classification head communication MB
4. total communication MB
5. cumulative communication MB
6. communication saving ratio compared with Standard LoRA



- classification head 
- FFA-LoRA / RoLoRA  LoRA adapter  Standard LoRA 
-  total communication ratio  classification head

==================================================
Checkpointing 
==================================================

 checkpointing

 global round



-  checkpoint_dir
-  method / alpha / round  checkpoint
-  metrics CSV
-  communication CSV
-  aggregation_bias CSV
-  client label distribution CSV
-  resume_from_checkpoint

checkpoint  base model  


- LoRA weights
- classification head weights
- method
- alpha
- round_idx
- config
- metrics so far
- communication logs
- aggregation bias logs
- random seed

checkpoint 

{
    "method": method,
    "alpha": alpha,
    "round_idx": round_idx,
    "global_state": global_state,
    "metrics": metrics_so_far,
    "communication_logs": communication_logs,
    "aggregation_bias_logs": aggregation_bias_logs,
    "config": config,
    "seed": seed
}



1. save_checkpoint()
2. load_checkpoint()
3. find_latest_checkpoint()
4. resume_experiment()

==================================================

==================================================

 quick mode  full mode

--------------------------------------------------
Quick mode
--------------------------------------------------

 Colab 



- alpha values = [0.5]
- methods = ["standard_lora", "ffa_lora", "rolora"]
- num_clients = 5
- clients_per_round = 5
- max_train_samples_per_client = 100  300
- max_eval_samples = 500
- global_rounds = 2  3
- local_epochs = 1
- batch_size = 1
- gradient_accumulation_steps = 4
- max_seq_length = 128
- bias_mode = "sampled_layers"
- max_bias_layers = 4

--------------------------------------------------
Full mode
--------------------------------------------------

 checkpointing



- alpha values = [10.0, 1.0, 0.5, 0.1]
- methods = ["standard_lora", "ffa_lora", "rolora"]
- num_clients = 5
- clients_per_round = 3  5
- max_train_samples_per_client = 500  1500
- max_eval_samples = 1000  3000
- global_rounds = 5  10
- local_epochs = 1
- batch_size = 1  2
- gradient_accumulation_steps = 4  8
- max_seq_length = 128  256
- bias_mode = "sampled_layers"
- max_bias_layers = 6  8

 CONFIG  quick / full mode

 quick mode Colab OOM  timeout

==================================================
 LoRA 
==================================================



- model_name = "Qwen/Qwen2.5-1.5B"
   1.5B sequence classification compatible model
- num_labels = 4
- load_in_4bit = True
- bnb_4bit_quant_type = "nf4"
- bnb_4bit_compute_dtype = torch.float16
- bnb_4bit_use_double_quant = True
- LoRA rank r = 8
- LoRA alpha = 16
- LoRA dropout = 0.05
- bias = "none"
- task_type = TaskType.SEQ_CLS

target_modules  Qwen 

- q_proj
- k_proj
- v_proj
- o_proj



- gate_proj
- up_proj
- down_proj

 attention projection modules VRAM 

==================================================

==================================================



Phase 1
Phase 2 Google Drive  checkpoint directory
Phase 3CONFIGrandom seed
Phase 4 AG News dataset
Phase 5 tokenizer 
Phase 6Dirichlet Non-IID split function
Phase 7client dataset / dataloader 
Phase 84-bit Sequence Classification model 
Phase 9LoRA config  PEFT model 
Phase 10classification head detection  requires_grad 
Phase 11FedAvg 
Phase 12communication cost 
Phase 13memory-safe layer-by-layer aggregation bias 
Phase 14local client training function
Phase 15global evaluation function
Phase 16checkpoint save / load / resume functions
Phase 17federated training loop
Phase 18run experiment for methods and alpha values
Phase 19 DataFrame 
Phase 20
Phase 21

==================================================

==================================================



1.  Python 
2.  markdown 
3.  docstring
4.  quick mode
5.  full mode  CONFIG
6.  logs  pandas DataFrame 
7.  logs  CSV
8.  global round  checkpoint  Google Drive
9.  OOM fallback 
   -  max_seq_length
   -  max_train_samples_per_client
   -  max_eval_samples
   -  LoRA rank
   -  target_modules
   -  max_bias_layers
10. 
    -  trade-off
    - Non-IID 
    - aggregation bias 
    - classification head 
    - FFA-LoRA  RoLoRA 
    - RoLoRA 

==================================================

==================================================

 sanity checks

1.  requires_grad=True 
2.  transmitted parameter names
3.  classification head  requires_grad=True
4.  base model weights  requires_grad=True
5.  FFA-LoRA  lora_A requires_grad=False
6.  RoLoRA  round  lora_A + classification head
7.  RoLoRA  round  lora_B + classification head
8.  communication cost  classification head
9.  aggregation bias  sampled layers  layer-by-layer 
10.  optimizer  client  local training 

 Local materialize  Delta W 


1. latex 
2. 
github
repo : git@github.com:jimmy01081122/DMLS-Project.git
