LoRA/FFLoRA/RoLoRA LoRA-based Federated Fine-tuning  Non-IID 
 PEFT  Python 

==================================================
 (RTX 3050 6GB VRAM, 15GB RAM)
==================================================
1. VRAM  (6GB)
2. `max_seq_length=128`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`  `8`
3. Optimizer  `bitsandbytes.optim.PagedAdamW8bit`  Optimizer  VRAM
4.  4-bit NF4 

==================================================

==================================================


--------------------------------------------------
 FedProx  ( Non-IID )
--------------------------------------------------
-  Client  `local_train`  L2 
- 
  1.  (`global_state`)  CPU
  2.  Loss  `requires_grad=True` 
  3.  `global_param`  GPU  `(local_param - global_param).pow(2).sum()`
  4.  `proximal_term`  `del global_param`  VRAM 
-  `mu` ( 0.01) FedAvg  FedProx 

--------------------------------------------------
 ()
--------------------------------------------------
-  `federated training loop` 
- 
  1.  `client_dropout_rate = 0.2` (20% )
  2.  Global Round  Clients
  3.  Client  Local Training  Client /
  4.  Client Server  Client 
  5.  RoLoRA  Validation Accuracy 

--------------------------------------------------
 ()
--------------------------------------------------
- 
- 
  1.  LoRA Rank Client  Rank Client 0,1  $r=4$Client 2,3,4  $r=8$
  2.  `fedavg_states` 
  3.  LoRA A  B 
  4.  Zero-padding  $r=4$  $r=8$  `torch.sum` 
  5.  (Classification Head) 

==================================================

==================================================
1.  `local_train_fedprox`  VRAM 
2.  `federated training loop`  Dropout 
3.  Zero-padding  `fedavg_heterogeneous_states` 
4.  Tensor  CPU  () Forward/Backward  FedProx Penalty  GPU

NOTE : 
1.  Local
2. 
3. 
4. /home/a/dmls/docs/report_en.tex  report_zh.md
   1. latex ( conference / journal )
   2. 
5. github
  repo : git@github.com:jimmy01081122/DMLS-Project.git
6. 
7. 1.5GB
