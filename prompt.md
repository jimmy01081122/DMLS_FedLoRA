本實驗目標是在資源受限環境下，在原實作LoRA/FFLoRA/RoLoRA基礎上，比較三種 LoRA-based Federated Fine-tuning 方法在 Non-IID 文本分類任務上的表現、通訊成本與聚合穩定性。
請扮演分散式機器學習與聯邦學習專家，基於我已完成的「單機模擬聯邦學習與 PEFT 比較專案」，保留原程式碼，額外創建並提供與執行後續「系統與演算法魯棒性驗證」的 Python 實作程式碼。

==================================================
一、硬體與環境嚴格限制 (RTX 3050 6GB VRAM, 15GB RAM)
==================================================
1. VRAM 極度受限 (6GB)。必須嚴格執行記憶體回收。
2. 訓練參數強制設定：`max_seq_length=128`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=4` 或 `8`。
3. Optimizer 強制規定：必須使用 `bitsandbytes.optim.PagedAdamW8bit` 以節省 Optimizer 狀態佔用的 VRAM。
4. 模型設定維持 4-bit NF4 量化。

==================================================
二、實作目標與階段要求
==================================================
請提供以下三個階段的實作邏輯與修改後的函數程式碼：

--------------------------------------------------
階段一：實作 FedProx 正則化 (解決高 Non-IID 偏移)
--------------------------------------------------
- 目標：在 Client 端的 `local_train` 迴圈中加入 L2 懲罰項。
- 記憶體安全要求：
  1. 全域模型權重 (`global_state`) 必須保存在 CPU。
  2. 計算 Loss 時，撰寫一個迴圈逐一取出 `requires_grad=True` 的參數。
  3. 將對應的 `global_param` 移至 GPU 計算 `(local_param - global_param).pow(2).sum()`。
  4. 加總至 `proximal_term` 後，立刻執行 `del global_param` 避免 VRAM 溢出。
- 輸入參數需增加 `mu` (例如 0.01)，並設計能切換 FedAvg 與 FedProx 的邏輯。

--------------------------------------------------
階段二：實作節點失效與延遲機制 (測試系統魯棒性)
--------------------------------------------------
- 目標：在 `federated training loop` 中模擬網路不穩定。
- 實作細節：
  1. 引入參數 `client_dropout_rate = 0.2` (20% 機率失效)。
  2. 在每個 Global Round 開始時，選定參與訓練的 Clients。
  3. 在 Client 完成 Local Training 並準備將權重加入聚合列表前，透過隨機數決定該 Client 是否「上傳超時/失敗」。
  4. 若失敗，則捨棄該 Client 本輪更新，Server 僅使用成功上傳的 Client 權重進行聚合。
  5. 特別記錄 RoLoRA 在經歷節點失效時的 Validation Accuracy 變化，觀察是否有崩潰現象。

--------------------------------------------------
階段三：動態秩分配與異質聚合 (測試通訊魯棒性)
--------------------------------------------------
- 目標：模擬不同運算能力的邊緣設備。
- 實作細節：
  1. 取消全域統一的 LoRA Rank。在建立 Client 虛擬狀態時，指派不同 Rank。例如 Client 0,1 使用 $r=4$；Client 2,3,4 使用 $r=8$。
  2. 修改伺服器端的 `fedavg_states` 函數。
  3. 當進行加權平均時，檢查傳入的 LoRA A 與 B 矩陣形狀。
  4. 實作 Zero-padding 機制：對於 $r=4$ 的權重，在其降維維度上補零擴充至 $r=8$ 的形狀，再與其他矩陣進行 `torch.sum` 加權聚合。
  5. 分類頭 (Classification Head) 維度不變，照常聚合。

==================================================
三、輸出要求
==================================================
1. 提供完整的 `local_train_fedprox` 函數程式碼，包含詳細的逐參數 VRAM 釋放邏輯。
2. 提供修改後的 `federated training loop` 程式碼片段，包含 Dropout 機制。
3. 提供支援 Zero-padding 的 `fedavg_heterogeneous_states` 函數程式碼。
4. 所有的 Tensor 操作若無必要，必須確保在 CPU 上進行 (例如聚合計算)，僅在訓練 Forward/Backward 與 FedProx Penalty 計算時使用 GPU。

NOTE : 
1. 程式需要適合在 Local上執行，不要只是寫程式碼，完成後執行，並自我審查邏輯正確性
2. 務必將所有有指標性的數據整理彙總，製成圖表或表格
3. 務必維護一份日誌與環境依賴設置文件
4. 務必額外寫出兩種研究報告，再整合入/home/a/dmls/docs/report_en.tex 和 report_zh.md
   1. 依照英文版latex 格式，排版依照學術期刊的報告(使用 conference / journal 模板)
   2. 中文版一般格式
5. 務必在每項重要更新節點前利用github做版本控制
  repo : git@github.com:jimmy01081122/DMLS-Project.git
6. 在作業中，如果有需要以軟體工程師的角度維護工作區結構
7. 如果有超過1.5GB檔案需要下載或方向不清楚，則先暫停並詢問我
