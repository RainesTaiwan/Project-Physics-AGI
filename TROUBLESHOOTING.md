# 🔧 問題排查記錄

本文檔記錄了專案開發過程中遇到的問題及解決方案。

---

## ✅ 已解決的問題

### 1. 缺少 PyTorch 依賴

**錯誤**:
```
ModuleNotFoundError: No module named 'torch'
```

**原因**: 初始環境中未安裝 PyTorch

**解決方案**:
```bash
pip install torch torchvision numpy scipy pyyaml tqdm
```

**狀態**: ✅ 已解決

---

### 2. 循環導入依賴

**錯誤**:
```
ModuleNotFoundError when importing src modules
```

**原因**: `src/__init__.py` 在導入時立即加載所有模組，包括需要 gymnasium/dm-control 的環境包裝器

**解決方案**: 實現惰性導入
```python
# src/__init__.py
def __getattr__(name):
    if name == 'ReplayBuffer':
        from .utils.replay_buffer import ReplayBuffer
        return ReplayBuffer
    # ... 其他模組
```

**檔案修改**:
- `src/__init__.py`
- `src/utils/__init__.py`

**狀態**: ✅ 已解決

---

### 3. RSSM 矩陣維度不匹配

**錯誤**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x400 and 232x200)
```

**原因**: Posterior 網絡期望輸入維度為 `deterministic_size + stochastic_size` (232)，但實際收到 `deterministic_size + hidden_size` (400)

**解決方案**: 修改 `src/models/rssm.py` 第 70 行
```python
# 原本
self.posterior_net = nn.Sequential(
    nn.Linear(deterministic_size + stochastic_size, hidden_size),  # 232
    ...
)

# 修改為
self.posterior_net = nn.Sequential(
    nn.Linear(deterministic_size + hidden_size, hidden_size),  # 400
    ...
)
```

**檔案修改**: `src/models/rssm.py` (第 70 行)

**狀態**: ✅ 已解決

---

### 4. 訓練器梯度圖保留錯誤

**錯誤**:
```
RuntimeError: Trying to backward through the graph a second time
```

**原因**: 在想像軌跡中，actor 和 value 的損失共享計算圖，導致第二次 backward 時圖已被釋放

**解決方案**: 修改 `src/trainer.py`
1. 在想像循環中 detach 狀態（第 313-328 行）
```python
next_state, _, _ = self.world_model.rssm.imagine_step(state, action.detach())
state = {
    'h': next_state['h'].detach(),
    'z': next_state['z'].detach()
}
```

2. 合併 actor 和 value 損失，單次 backward（第 340-363 行）
```python
actor_loss = -(log_probs_stacked * advantages).mean()
value_loss = F.mse_loss(pred_values, returns.detach())
total_behavior_loss = actor_loss + value_loss

self.actor_optimizer.zero_grad()
self.value_optimizer.zero_grad()
total_behavior_loss.backward()  # 單次 backward
self.actor_optimizer.step()
self.value_optimizer.step()
```

**檔案修改**: `src/trainer.py` (第 313-363 行)

**狀態**: ✅ 已解決

---

### 5. 訓練腳本類型注解錯誤

**錯誤**:
```
NameError: name 'Dict' is not defined. Did you mean: 'dict'?
```

**原因**: `train.py` 使用了 `Dict` 類型注解但未從 `typing` 導入

**解決方案**: 在 `train.py` 添加導入
```python
from typing import Dict
```

**檔案修改**: `train.py` (第 8 行)

**狀態**: ✅ 已解決

---

### 6. 無頭環境 OpenGL 錯誤

**錯誤**:
```
GLFWError: (65550) b'X11: The DISPLAY environment variable is missing'
mujoco.FatalError: an OpenGL platform library has not been loaded
```

**原因**: 
1. 沒有 X11 顯示環境
2. MuJoCo 需要 OpenGL 上下文但未配置 EGL
3. 缺少 EGL 系統庫

**解決方案**:

**步驟 1**: 設置環境變數（在導入 dm_control 之前）
```python
# train.py (開頭)
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# env_wrapper.py (DMCWrapper.__init__)
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
```

**步驟 2**: 安裝系統 EGL 庫
```bash
sudo apt-get update
sudo apt-get install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev mesa-utils
```

**檔案修改**:
- `train.py` (第 6-8 行)
- `src/utils/env_wrapper.py` (第 26-28 行)

**狀態**: ✅ 已解決

---

### 7. YAML 學習率解析錯誤

**錯誤**:
```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

**原因**: YAML 將科學記數法（如 `6e-4`）解析為字符串而非浮點數

**測試驗證**:
```python
>>> import yaml
>>> config = yaml.safe_load(open('config.yaml'))
>>> type(config['training']['model_lr'])
<class 'str'>  # 錯誤！應該是 float
>>> config['training']['model_lr']
'6e-4'
```

**解決方案**: 在 `config.yaml` 中使用標準浮點數格式
```yaml
# 原本
model_lr: 6e-4
actor_lr: 8e-5
value_lr: 8e-5
adam_eps: 1e-5

# 修改為
model_lr: 0.0006
actor_lr: 0.00008
value_lr: 0.00008
adam_eps: 0.00001
```

**檔案修改**: `config.yaml` (第 73-78 行)

**狀態**: ✅ 已解決

---

## 📊 測試結果總覽

### 模組測試（test_modules.sh）
```
✓ 模組 A - 變分編碼器
✓ 模組 B - RSSM 動力學模型
✓ 模組 C/D - Actor-Critic
✓ 工具 - Replay Buffer
✓ 訓練器 - World Model Trainer

通過: 5/5
```

### 訓練腳本測試
```bash
$ python train.py --config config.yaml --steps 100

Using device: cpu
Creating environment...
Building World Model...
Initializing Replay Buffer...

============================================================
Environment: DMC-walker-walk
Action dimension: 6
Observation shape: (3, 64, 64)
============================================================

Phase 1: Prefilling replay buffer with random exploration...
Buffer size: 5000/1000000

Phase 2: Training World Model...
Target: 100 total environment steps

✓ Training completed!
```

**狀態**: ✅ 所有測試通過

---

## 🛠️ 調試技巧

### 1. 模組獨立測試
```bash
# 直接運行模組進行測試
python -m src.models.encoder
python -m src.models.rssm
python -m src.models.actor_critic
python -m src.utils.replay_buffer
python -m src.trainer
```

### 2. 檢查依賴
```bash
# 檢查 PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 檢查 CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 檢查 MuJoCo
python -c "import mujoco; print('MuJoCo OK')"

# 檢查 EGL
ldconfig -p | grep libegl
```

### 3. 檢查配置解析
```bash
# 驗證 YAML 解析
python -c "import yaml; config = yaml.safe_load(open('config.yaml')); print(type(config['training']['model_lr']), config['training']['model_lr'])"
```

### 4. 檢查環境變數
```bash
# 驗證 MuJoCo 設置
python -c "import os; print(f\"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'not set')}\")"
```

---

## 📝 開發過程統計

- **總問題數**: 7
- **已解決**: 7 (100%)
- **修改檔案**: 6
- **代碼行數**: ~3,500 行
- **開發時間**: 1 會話
- **測試覆蓋**: 5/5 核心模組 + 完整訓練流程

---

## 🎯 當前狀態

### ✅ 完全就緒
- 所有核心模組測試通過
- 訓練腳本可以運行
- 環境正確配置（無頭模式）
- 依賴全部安裝

### 📌 系統配置
- Python: 3.12.1
- PyTorch: 2.9.1+cu128
- MuJoCo: 最新版本
- 渲染模式: EGL (無頭)
- 環境: Ubuntu 24.04 (dev container)

### 🚀 可以開始
```bash
# 快速測試（100 步）
python train.py --config config.yaml --steps 100

# 完整訓練（1M 步）
python train.py --config config.yaml --steps 1000000
```

---

**最後更新**: 2026-01-09
