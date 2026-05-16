請扮演分散式機器學習、聯邦學習、PEFT、LoRA 與大型語言模型微調領域的研究專家，協助我設計並實作一個可在RTX 3050 執行的單機模擬聯邦學習實驗。
本實驗目標是在資源受限環境下，比較三種 LoRA-based Federated Fine-tuning 方法在 Non-IID 文本分類任務上的表現、通訊成本與聚合穩定性。

==================================================
一、硬體與環境限制
==================================================

請假設執行環境為：

- 單張 NVIDIA 3050
- VRAM自行查詢
- 約 15GB 系統 RAM
- 可能發生 session timeout
- 不允許同時載入多個 client model
- 必須採用 sequential client simulation
- 必須積極釋放 GPU / CPU 記憶體
- 必須支援 checkpointing，以便中途中斷後恢復

請務必遵守以下限制：

1. 基礎模型必須使用 bitsandbytes 4-bit NF4 量化。
2. 不得 full fine-tune base model。
3. 不得同時在 GPU 上保留多個 client model。
4. client 權重更新必須搬到 CPU RAM 暫存。
5. FedAvg 必須在 CPU 上完成。
6. 每個 client local training 結束後必須釋放不必要物件：
   - del dataloader 或 batch cache
   - del optimizer
   - del loss
   - torch.cuda.empty_cache()
   - gc.collect()

==================================================
二、核心任務設定
==================================================

請使用：

- Dataset：AG News
- Task：4-class text classification
- Model architecture：Sequence Classification
- Base model：Qwen2.5-1.5B 或其他 1.5B 等級 decoder-only model
- 載入方式：AutoModelForSequenceClassification
- num_labels = 4
- Quantization：4-bit NF4
- PEFT：LoRA

請注意：

本實驗明確採用 Sequence Classification，而不是 Next-Token Prediction / Causal LM label generation。

原因是 AG News 是標準四分類任務，Sequence Classification 可以直接輸出 4 維 logits，方便計算 accuracy、macro-F1、validation loss、FedAvg 權重集合與通訊成本。

模型結構應視為：

4-bit quantized base LLM backbone
+ LoRA adapters
+ trainable classification head

其中 classification head 是隨機初始化的分類矩陣，不屬於 LoRA 參數，但必須：

1. requires_grad=True
2. 參與每個 client 的 local training
3. 參與 server-side FedAvg
4. 納入 communication cost
5. 納入 checkpoint saving / loading

請勿只傳輸 LoRA 權重。

==================================================
三、欲比較的方法
==================================================

請比較以下三種方法：

--------------------------------------------------
1. Standard LoRA
--------------------------------------------------

設定：

- LoRA A 可訓練
- LoRA B 可訓練
- classification head 可訓練

每輪 client 上傳：

- lora_A
- lora_B
- classification head

Server 使用 weighted FedAvg 聚合上述權重。

--------------------------------------------------
2. FFA-LoRA
--------------------------------------------------

設定：

- 初始化後凍結 LoRA A
- 僅訓練 LoRA B
- classification head 可訓練

每輪 client 上傳：

- lora_B
- classification head

Server 使用 weighted FedAvg 聚合上述權重。

請注意：

FFA-LoRA 的「通訊減半」僅嚴格適用於 LoRA adapter 部分。  
因為 classification head 也必須傳輸，所以總通訊量不一定剛好是 Standard LoRA 的 50%。

請輸出：

FFA total communication ratio =
(size(lora_B) + size(classification_head))
/
(size(lora_A) + size(lora_B) + size(classification_head))

--------------------------------------------------
3. RoLoRA
--------------------------------------------------

設定：

- 採用交替最佳化
- 奇數 global round：凍結 B，只訓練 A
- 偶數 global round：凍結 A，只訓練 B
- classification head 每一輪都可訓練

奇數 round client 上傳：

- lora_A
- classification head

偶數 round client 上傳：

- lora_B
- classification head

Server 使用 weighted FedAvg 聚合當輪上傳權重。

請注意：

RoLoRA 的「通訊減半」也僅嚴格適用於 LoRA adapter 部分。  
總通訊量需加上 classification head，因此不一定剛好是 Standard LoRA 的 50%。

==================================================
四、研究問題
==================================================

請圍繞以下三個研究問題設計實驗。

==================================================
研究問題一：通訊成本 vs 準確率
==================================================

請回答：

在相同聯邦訓練設定下，Standard LoRA、FFA-LoRA 與 RoLoRA 的驗證準確率、macro-F1、validation loss 與通訊成本有何差異？

請實作：

1. 每個 global round 後，在 validation / test set 上評估：
   - Accuracy
   - Macro-F1
   - Validation loss

2. 每個 round 計算：
   - 單輪通訊量 MB
   - 累積通訊量 MB
   - trainable parameters
   - transmitted parameters
   - LoRA-only communication MB
   - total communication MB including classification head

3. 請輸出以下圖表：
   - Accuracy vs Global Round
   - Macro-F1 vs Global Round
   - Validation Loss vs Global Round
   - Cumulative Communication MB vs Accuracy
   - Cumulative Communication MB vs Macro-F1
   - Per-round Communication MB by Method

4. 最後輸出 summary table，包含：
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

請特別驗證：

- FFA-LoRA 的 LoRA adapter 通訊量是否約為 Standard LoRA 的 50%。
- RoLoRA 的 LoRA adapter 通訊量是否約為 Standard LoRA 的 50%。
- 加上 classification head 後，總通訊量是否仍接近 50%。
- 通訊量降低是否造成 accuracy 或 macro-F1 顯著下降。

==================================================
研究問題二：Non-IID 程度對三種方法的影響
==================================================

請回答：

當資料異質性增加時，三種方法的收斂穩定性與最終表現如何變化？

請使用 Dirichlet distribution 建立 5 個 virtual clients 的 Non-IID 資料分布。

請測試以下 Dirichlet alpha：

- alpha = 10.0，代表近似 IID
- alpha = 1.0，代表中度 Non-IID
- alpha = 0.5，代表高度 Non-IID
- alpha = 0.1，代表極端 Non-IID

請實作：

1. Dirichlet Non-IID split function：
   - input：dataset labels, num_clients, alpha, seed
   - output：client_indices
   - 確保每個 client 至少有 min_samples
   - 若某 client 樣本過少，需重新抽樣或修正分配

2. 對每個 alpha 輸出 client label distribution table：
   - row：client
   - columns：World, Sports, Business, Sci/Tech
   - values：樣本數
   - 額外輸出各 client 的 label proportion

3. 對每個 alpha 與每種方法進行聯邦訓練。

4. 記錄每種方法在不同 alpha 下的：
   - Final Accuracy
   - Final Macro-F1
   - Best Accuracy
   - Best Macro-F1
   - Final Validation Loss
   - Accuracy standard deviation across rounds
   - Macro-F1 standard deviation across rounds
   - Total Communication MB
   - Mean Aggregation Bias

5. 請輸出以下圖表：
   - Alpha vs Final Accuracy
   - Alpha vs Final Macro-F1
   - Alpha vs Total Communication MB
   - Alpha vs Mean Aggregation Bias
   - Accuracy curves under different alpha values
   - Macro-F1 curves under different alpha values

請在結果分析中討論：

- Standard LoRA 是否在 alpha 較小時更容易震盪？
- FFA-LoRA 是否在高 Non-IID 下更穩定但表達能力較弱？
- RoLoRA 是否能在高 Non-IID 下取得比 FFA-LoRA 更好的準確率，同時維持較低通訊成本？
- 哪個方法在 alpha = 0.1 或 alpha = 0.5 時最具 robustness？

==================================================
研究問題三：LoRA 在 FedAvg 下的 Aggregation Bias
==================================================

請回答：

Standard LoRA 在聯邦學習中分別平均 A 與 B 是否會造成 aggregation bias？  
FFA-LoRA 與 RoLoRA 是否能降低此問題？

對每個 LoRA layer，LoRA update 定義為：

Delta W_k = B_k A_k

其中 k 代表 client。

Standard LoRA 的 FedAvg 實際聚合方式為：

A_avg = average(A_k)
B_avg = average(B_k)
Delta W_fedavg = B_avg A_avg

但理想 client update 平均為：

Delta W_ideal = average(B_k A_k)

Aggregation Bias 定義為：

Aggregation Bias =
|| Delta W_ideal - Delta W_fedavg ||_F
/
|| Delta W_ideal ||_F

請注意：

絕對不可一次 materialize 所有 layers 的完整 Delta W。  
這會導致 Colab RAM 或 GPU VRAM OOM。

必須採用 layer-by-layer 計算與即時計算釋放。

請實作 memory-safe aggregation bias function：

1. 每次只處理一個 LoRA layer。
2. 從所有 client state 中取出該 layer 的 A_k 與 B_k。
3. 在 CPU 或短暫 GPU 上計算該層的 B_k @ A_k。
4. 計算該層的 aggregation bias scalar。
5. 將 scalar 加入 list。
6. 立刻 del 該層所有大型 tensor。
7. 執行：
   - gc.collect()
   - torch.cuda.empty_cache()
8. 再處理下一層。

請支援兩種模式：

--------------------------------------------------
bias_mode = "sampled_layers"
--------------------------------------------------

預設使用 sampled_layers，

建議抽樣：

- 前 2 個 LoRA layers
- 中間 2 個 LoRA layers
- 最後 2 個 LoRA layers

或最多 sample 6 至 8 個 LoRA layers。

--------------------------------------------------
bias_mode = "all_layers"
--------------------------------------------------

僅在記憶體足夠時使用。

請對三種方法分別計算：

1. Standard LoRA aggregation bias
2. FFA-LoRA aggregation bias
3. RoLoRA aggregation bias

對 FFA-LoRA：

因為 A 是固定共享的 A_0，請驗證：

average(B_k A_0) 是否接近 average(B_k) A_0

預期 aggregation bias 應接近 0 或顯著低於 Standard LoRA。

對 RoLoRA：

若該 round 只更新 A，則 B 應保持全域一致。  
若該 round 只更新 B，則 A 應保持全域一致。

請驗證其 aggregation bias 是否低於 Standard LoRA。

請輸出以下圖表：

- Aggregation Bias vs Global Round
- Aggregation Bias vs Accuracy
- Aggregation Bias vs Macro-F1
- Aggregation Bias under different Dirichlet alpha values

請輸出 summary table：

- Method
- Alpha
- Mean Aggregation Bias
- Final Aggregation Bias
- Final Accuracy
- Final Macro-F1
- Total Communication MB

請在討論中分析：

- Standard LoRA 的 aggregation bias 是否隨 Non-IID 程度上升而增加？
- Aggregation bias 是否與 accuracy drop 或 training instability 有關？
- FFA-LoRA 是否因固定 A 而降低 bias，但犧牲模型表達能力？
- RoLoRA 是否能兼顧低 aggregation bias 與較佳表達能力？

==================================================
五、Optimizer 狀態處理要求
==================================================

請特別注意 RoLoRA 與 optimizer state 的衝突問題。

由於 AdamW、paged_adamw_8bit 等 optimizer 含有 momentum / variance 狀態，RoLoRA 在交替凍結與解凍 A/B 時，如果沿用舊 optimizer state，可能使 optimizer state 與當前 global weights 脫節。

因此請強制規定：

每一次 client local training 開始時，都必須重新初始化 optimizer。

也就是：

- 每個 client 都建立新的 optimizer。
- 每個 global round 都建立新的 optimizer。
- 不得跨 client 重用 optimizer。
- 不得跨 round 重用 optimizer。
- client 訓練結束後 del optimizer。

local_train function 中必須包含：

1. set_trainable_params(model, method, round_idx)
2. trainable_params = [p for p in model.parameters() if p.requires_grad]
3. optimizer = create_optimizer(trainable_params)
4. local training
5. extract transmitted state
6. del optimizer
7. gc.collect()
8. torch.cuda.empty_cache()

==================================================
六、Classification Head 處理要求
==================================================

因為 AutoModelForSequenceClassification 會有獨立分類頭，因此請實作 robust classification head detection function。

例如：

is_classification_head(name)

需能偵測可能名稱：

- score.weight
- score.bias
- classifier.weight
- classifier.bias
- classification_head.*
- 任務模型中的其他分類輸出層名稱

但不得誤把 LoRA 參數納入 classification head。

建議邏輯：

- 如果 name 包含 "lora_"，則不是 classification head。
- 如果 name 包含 "score"、"classifier"、"classification_head"，則視為 classification head。
- 請在模型載入後印出所有 trainable / transmitted parameter names，供使用者檢查。

classification head 在三種方法中都必須：

1. requires_grad=True
2. 參與 local training
3. 參與 FedAvg
4. 納入 communication cost
5. 納入 checkpoint
6. 在 resume checkpoint 時正確載入

==================================================
七、權重萃取、載入與 FedAvg 要求
==================================================

請實作以下函數：

--------------------------------------------------
1. should_train_param(name, method, round_idx)
--------------------------------------------------

判斷該參數是否在 local training 中 requires_grad=True。

規則：

Standard LoRA：
- train lora_A
- train lora_B
- train classification head

FFA-LoRA：
- freeze lora_A
- train lora_B
- train classification head

RoLoRA：
- odd round：train lora_A + classification head
- even round：train lora_B + classification head

--------------------------------------------------
2. should_transmit_param(name, method, round_idx)
--------------------------------------------------

判斷該參數是否需要 client 上傳給 server。

規則：

Standard LoRA：
- transmit lora_A
- transmit lora_B
- transmit classification head

FFA-LoRA：
- transmit lora_B
- transmit classification head

RoLoRA：
- odd round：transmit lora_A + classification head
- even round：transmit lora_B + classification head

--------------------------------------------------
3. extract_transmitted_state(model, method, round_idx)
--------------------------------------------------

將需要傳輸的參數 detach、clone 並移至 CPU。

不得保存不需要傳輸的 base model weights。

--------------------------------------------------
4. load_global_state(model, global_state)
--------------------------------------------------

將 server 端聚合後的 global_state 載入模型。

只更新 LoRA 與 classification head。

--------------------------------------------------
5. fedavg_states(client_states, client_sizes)
--------------------------------------------------

在 CPU 上執行 weighted FedAvg。

公式：

global_weight =
sum_k (n_k / total_n) * client_weight_k

其中 n_k 為 client k 的資料量。

==================================================
八、通訊成本計算要求
==================================================

請實作 communication cost function。

每個 parameter 的通訊成本以實際 tensor element 數與 dtype size 估計。

請分別計算：

1. LoRA A communication MB
2. LoRA B communication MB
3. classification head communication MB
4. total communication MB
5. cumulative communication MB
6. communication saving ratio compared with Standard LoRA

請注意：

- classification head 必須納入總通訊成本。
- FFA-LoRA / RoLoRA 的 LoRA adapter 部分可能約為 Standard LoRA 的一半。
- 但 total communication ratio 需額外加上 classification head。

==================================================
九、Checkpointing 與恢復訓練要求
==================================================

請務必實作 checkpointing。

每完成一個 global round，就將結果儲存。

請支援：

- 設定 checkpoint_dir
- 每個 method / alpha / round 儲存一個 checkpoint
- 儲存 metrics CSV
- 儲存 communication CSV
- 儲存 aggregation_bias CSV
- 儲存 client label distribution CSV
- 支援 resume_from_checkpoint

checkpoint 不得保存完整 base model。  
只保存：

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

checkpoint 格式建議：

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

請提供：

1. save_checkpoint()
2. load_checkpoint()
3. find_latest_checkpoint()
4. resume_experiment()

==================================================
十、實驗模式
==================================================

請提供 quick mode 與 full mode。

--------------------------------------------------
Quick mode
--------------------------------------------------

用於確認 Colab 可執行，不追求最終研究品質。

設定：

- alpha values = [0.5]
- methods = ["standard_lora", "ffa_lora", "rolora"]
- num_clients = 5
- clients_per_round = 5
- max_train_samples_per_client = 100 到 300
- max_eval_samples = 500
- global_rounds = 2 或 3
- local_epochs = 1
- batch_size = 1
- gradient_accumulation_steps = 4
- max_seq_length = 128
- bias_mode = "sampled_layers"
- max_bias_layers = 4

--------------------------------------------------
Full mode
--------------------------------------------------

用於正式實驗，但必須依賴 checkpointing，不能假設一次跑完。

設定：

- alpha values = [10.0, 1.0, 0.5, 0.1]
- methods = ["standard_lora", "ffa_lora", "rolora"]
- num_clients = 5
- clients_per_round = 3 或 5
- max_train_samples_per_client = 500 到 1500
- max_eval_samples = 1000 到 3000
- global_rounds = 5 到 10
- local_epochs = 1
- batch_size = 1 或 2
- gradient_accumulation_steps = 4 或 8
- max_seq_length = 128 或 256
- bias_mode = "sampled_layers"
- max_bias_layers = 6 到 8

請在程式中用 CONFIG 控制 quick / full mode。

預設請使用 quick mode，避免 Colab OOM 或 timeout。

==================================================
十一、模型與 LoRA 設定
==================================================

請使用以下預設設定：

- model_name = "Qwen/Qwen2.5-1.5B"
  或其他可用的 1.5B sequence classification compatible model
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

target_modules 請根據 Qwen 類模型設定，例如：

- q_proj
- k_proj
- v_proj
- o_proj

如果記憶體允許，也可加入：

- gate_proj
- up_proj
- down_proj

但預設為 attention projection modules，以降低 VRAM 壓力。

==================================================
十二、程式區塊要求
==================================================

請將實作程式碼的流程分成以下階段：

Phase 1：安裝套件與環境檢查
Phase 2：掛載 Google Drive 與設定 checkpoint directory
Phase 3：匯入套件、CONFIG、random seed
Phase 4：載入 AG News dataset
Phase 5：建立 tokenizer 與資料前處理
Phase 6：Dirichlet Non-IID split function
Phase 7：client dataset / dataloader 建立函數
Phase 8：4-bit Sequence Classification model 載入
Phase 9：LoRA config 與 PEFT model 建立
Phase 10：classification head detection 與 requires_grad 控制函數
Phase 11：權重萃取、載入、FedAvg 函數
Phase 12：communication cost 計算函數
Phase 13：memory-safe layer-by-layer aggregation bias 計算函數
Phase 14：local client training function
Phase 15：global evaluation function
Phase 16：checkpoint save / load / resume functions
Phase 17：federated training loop
Phase 18：run experiment for methods and alpha values
Phase 19：結果 DataFrame 彙整
Phase 20：圖表輸出
Phase 21：研究結論撰寫模板

==================================================
十三、輸出要求
==================================================

請最後輸出：

1. 完整可執行 Python 程式碼。
2. 每個區塊前都有清楚 markdown 標題。
3. 所有重要函數都有 docstring。
4. 程式預設使用 quick mode。
5. 提供切換 full mode 的 CONFIG。
6. 所有 logs 都以 pandas DataFrame 儲存。
7. 所有 logs 都可輸出成 CSV。
8. 每個 global round 後自動 checkpoint 到 Google Drive。
9. 若發生 OOM，請提供 fallback 建議，例如：
   - 降低 max_seq_length
   - 降低 max_train_samples_per_client
   - 降低 max_eval_samples
   - 降低 LoRA rank
   - 減少 target_modules
   - 減少 max_bias_layers
10. 最後輸出研究分析模板，內容包含：
    - 通訊成本與準確率 trade-off
    - Non-IID 程度對方法穩定性的影響
    - aggregation bias 與模型表現的關係
    - classification head 對總通訊成本的影響
    - FFA-LoRA 與 RoLoRA 是否真的具有通訊優勢
    - RoLoRA 是否在低通訊量下保留較好的表達能力

==================================================
十四、重要安全檢查
==================================================

請在程式中加入以下 sanity checks：

1. 印出所有 requires_grad=True 的參數名稱。
2. 印出所有 transmitted parameter names。
3. 確認 classification head 有 requires_grad=True。
4. 確認 base model weights 沒有 requires_grad=True。
5. 確認 FFA-LoRA 中 lora_A requires_grad=False。
6. 確認 RoLoRA 奇數 round 只訓練 lora_A + classification head。
7. 確認 RoLoRA 偶數 round 只訓練 lora_B + classification head。
8. 確認 communication cost 包含 classification head。
9. 確認 aggregation bias 使用 sampled layers 或 layer-by-layer 計算。
10. 確認 optimizer 每個 client 每次 local training 都重新初始化。

務必使程式適合在 Local上執行，並避免任何會一次 materialize 全模型 Delta W 的操作。
務必將所有有指標性的數據整理彙總，製成圖表或表格
務必寫出兩種研究報告
1. 依照英文版latex 格式，排版依照學術期刊的報告
2. 中文版一般格式
務必在每項重要更新節點前利用github做版本控制
repo : git@github.com:jimmy01081122/DMLS-Project.git
