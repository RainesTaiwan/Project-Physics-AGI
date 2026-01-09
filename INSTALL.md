# 安裝指南

## 📦 依賴安裝

### 方法 1：使用 pip（推薦）

```bash
# 安裝核心依賴
pip install torch torchvision numpy scipy

# 安裝 RL 相關
pip install gymnasium

# 安裝可選依賴（用於完整功能）
pip install mujoco dm-control
pip install tensorboard wandb
pip install pyyaml tqdm matplotlib pillow
```

### 方法 2：從 requirements.txt 安裝

```bash
pip install -r requirements.txt
```

## 🧪 驗證安裝

運行測試腳本：

```bash
# 給腳本執行權限
chmod +x test_modules.sh

# 運行測試
./test_modules.sh
```

或者手動測試各個模組：

```bash
# 測試編碼器
python -m src.models.encoder

# 測試 RSSM
python -m src.models.rssm

# 測試 Actor-Critic
python -m src.models.actor_critic

# 測試 Replay Buffer
python -m src.utils.replay_buffer
```

## ⚠️ 常見問題

### 1. CUDA 不可用

如果沒有 GPU，系統會自動使用 CPU。要檢查 CUDA：

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

### 2. MuJoCo 安裝失敗

MuJoCo 需要額外的系統依賴：

**Ubuntu/Debian:**
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

**macOS:**
```bash
brew install glfw
```

### 3. dm-control 安裝失敗

確保先安裝 MuJoCo：

```bash
pip install mujoco
pip install dm-control
```

## 🎯 最小依賴配置

如果只想測試核心功能（不需要環境模擬）：

```bash
pip install torch numpy pyyaml
```

這足以運行模組測試，但無法進行完整訓練。

## 🚀 完整訓練需求

要運行完整的訓練流程，需要：

1. ✅ PyTorch >= 2.0
2. ✅ MuJoCo >= 2.3
3. ✅ dm-control >= 1.0
4. ✅ Gymnasium >= 0.28
5. 💡 CUDA 推薦但非必須

## 📊 性能建議

- **CPU only**: 可運行但訓練很慢（~10x slower）
- **GPU**: 推薦 NVIDIA GPU with >= 8GB VRAM
- **最佳配置**: NVIDIA A100/H100 with 40GB+ VRAM

## 🔧 Docker 選項（可選）

如果遇到安裝問題，可使用 Docker：

```bash
# TODO: 提供 Dockerfile
docker build -t physics-agi .
docker run --gpus all -it physics-agi
```

## 📞 需要幫助？

如果安裝遇到問題：

1. 檢查 Python 版本 >= 3.8
2. 更新 pip: `pip install --upgrade pip`
3. 創建全新虛擬環境
4. 提交 Issue 並附上錯誤信息
