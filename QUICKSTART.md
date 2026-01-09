# 🚀 快速開始指南

## ✅ 系統狀態

**所有核心模組測試通過！訓練腳本已就緒！**

```
✓ 模組 A - 變分編碼器
✓ 模組 B - RSSM 動力學模型  
✓ 模組 C/D - Actor-Critic
✓ 工具 - Replay Buffer
✓ 訓練器 - World Model Trainer
✓ 訓練腳本 - 完整訓練流程
```

**最新修復**:
- ✅ 添加 `Dict` 類型導入
- ✅ 配置 EGL 無頭渲染
- ✅ 安裝 Mesa OpenGL 庫
- ✅ 修復 YAML 學習率解析

---

## 📦 已安裝的依賴

核心依賴已安裝：
- ✅ PyTorch 2.9.1 (with CUDA 12.8)
- ✅ NumPy, SciPy, PyYAML, tqdm
- ✅ MuJoCo, DeepMind Control, Gymnasium
- ✅ Mesa EGL (無頭渲染支援)

---

## 🎯 下一步操作

### ✅ 環境已完全配置！

系統已準備好進行訓練：
- ✅ 所有核心依賴已安裝
- ✅ MuJoCo 環境已配置（無頭模式）
- ✅ 訓練腳本可以直接運行

---

## 🧪 測試模組

```bash
# 運行所有測試
./test_modules.sh

# 或單獨測試
python -m src.models.encoder      # 測試編碼器
python -m src.models.rssm         # 測試 RSSM
python -m src.models.actor_critic # 測試 Actor-Critic
python -m src.utils.replay_buffer # 測試緩衝區
python -m src.trainer             # 測試訓練器
```

---

## 🎮 開始訓練

### 1. 快速測試訓練（推薦先試）

```bash
# 100 步快速測試（約 1-2 分鐘，CPU）
python train.py --config config.yaml --steps 100
```

### 2. 短期訓練測試

```bash
# 10K 步測試（約 30-60 分鐘，CPU；5-10 分鐘，GPU）
python train.py --config config.yaml --steps 10000
```

### 3. 完整訓練

```bash
# 1M 步完整訓練（需要幾小時到幾天，取決於硬件）
# ⚠️ 注意：在 CPU 上會非常慢，建議使用 GPU
python train.py --config config.yaml --steps 1000000
```

### 4. 監控訓練（可選）

```bash
# 安裝 TensorBoard（如果尚未安裝）
pip install tensorboard

# 在另一個終端運行
tensorboard --logdir logs/

# 然後在瀏覽器打開 http://localhost:6006
```

---

## 📊 預期表現

### 訓練階段

**Dynamics Learning (物理學習)**:
- `reconstruction_loss` 應該下降到 < 0.1
- `kl_loss` 應該從 >10 降到 3-5
- 當 KL 趨近 0 時，AI 理解了物理法則

**Behavior Learning (策略學習)**:
- `episode_reward` 應該持續上升
- `actor_loss` 和 `value_loss` 應該穩定

### 訓練時間估計

| 硬件 | 1M steps 時間 |
|------|--------------|
| NVIDIA A100 | 8-12 小時 |
| NVIDIA RTX 3090 | 15-20 小時 |
| NVIDIA GTX 1080 | 1-2 天 |
| CPU only | 3-5 天（不推薦）|

---

## 🎨 可視化（訓練後）

```bash
# 可視化重建
python visualize.py --config config.yaml --checkpoint <path> --mode reconstruction

# 可視化想像軌跡
python visualize.py --config config.yaml --checkpoint <path> --mode imagination

# 可視化潛在空間
python visualize.py --config config.yaml --checkpoint <path> --mode latent
```

---

## ⚙️ 配置調整

編輯 `config.yaml` 來調整：

```yaml
# 環境選擇
environment:
  name: "DMC-walker-walk"     # 或 "DMC-cheetah-run", "DMC-cartpole-swingup"
  
# 關鍵超參數
rssm:
  imagination_horizon: 15     # 想像未來步數
  free_nats: 3.0              # KL 最小值
  
training:
  model_lr: 6e-4              # World Model 學習率
  actor_lr: 8e-5              # Actor 學習率
  batch_size: 50              # Batch 大小
```

---

## 🐛 常見問題

### Q: 測試失敗？
**A**: 運行 `./test_modules.sh` 查看具體錯誤。通常是缺少依賴。

### Q: 訓練腳本報錯 "NameError: name 'Dict' is not defined"?
**A**: 已修復！確保使用最新版本的 `train.py`。

### Q: OpenGL 或 EGL 錯誤？
**A**: 已安裝 Mesa EGL 支援！系統已配置無頭渲染。

### Q: YAML 配置錯誤 "TypeError: '<=' not supported"?
**A**: 已修復！學習率現在使用正確的浮點數格式。

### Q: CUDA out of memory?
**A**: 
1. 減少 `batch_size`（在 config.yaml）
2. 減少 `sequence_length`
3. 減少 `imagination_horizon`

### Q: 訓練很慢？
**A**:
1. 確認使用 GPU：`python -c "import torch; print(torch.cuda.is_available())"`
2. CPU 訓練會非常慢，建議使用 GPU 或減少訓練步數
3. 減少 `train_steps`（每次收集後的梯度更新次數）
4. 增加 `train_every`（減少訓練頻率）

### Q: KL 散度不下降？
**A**:
1. 檢查 `free_nats` 是否太大
2. 增加 `kl_weight`
3. 確保 RSSM 有足夠容量

---

## 📚 學習資源

### 項目文檔
- [README.md](README.md) - 主文檔和快速開始
- [TECHNICAL.md](TECHNICAL.md) - 技術細節和數學推導
- [INSTALL.md](INSTALL.md) - 詳細安裝指南
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 完整項目總覽

### 代碼導航
1. **核心模型**: `src/models/` - 查看各個模組的實現
2. **訓練器**: `src/trainer.py` - 理解訓練流程
3. **主腳本**: `train.py` - 完整訓練流程
4. **配置**: `config.yaml` - 所有超參數

---

## 🎯 推薦學習路徑

### 初學者
1. ✅ 運行 `./test_modules.sh` 確保所有模組工作
2. 📖 閱讀 `README.md` 理解系統架構
3. 🔍 查看 `src/models/encoder.py` 理解編碼器
4. 🧠 查看 `src/models/rssm.py` 理解動力學模型

### 進階用戶
1. 🎮 安裝環境並運行快速訓練測試
2. 📊 使用 TensorBoard 監控訓練
3. ⚙️ 調整 `config.yaml` 實驗不同設定
4. 📝 閱讀 `TECHNICAL.md` 理解數學原理

### 研究者
1. 📖 閱讀完整項目文檔
2. 🔬 修改模型架構進行實驗
3. 📊 分析訓練曲線和可視化結果
4. 🚀 擴展到新環境或任務

---

## ✅ 系統已就緒

**當前狀態**: 所有核心模組測試通過 ✅

**可以開始**:
- ✅ 研究代碼
- ✅ 修改模型
- ✅ 運行測試
- ⏳ 訓練模型（需要安裝環境）

---

**有問題？** 查看文檔或提交 Issue！
