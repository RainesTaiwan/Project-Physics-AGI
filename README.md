# Project Physics-AGI

## ✅ 狀態：完全就緒 | 訓練腳本可運行

**最新更新** (2026-01-09):
- ✅ 修復所有依賴問題
- ✅ 配置無頭渲染（EGL）
- ✅ 訓練腳本完全可用
- ✅ 所有 5/5 模組測試通過

---

## 🧠 系統架構：Model-Based Reinforcement Learning (MBRL)

這是一個完整的 World Model 實現,採用 **物理感知 AI** 架構，能夠：
- 🎯 **理解物理法則**（而非死記硬背）
- 🔮 **在腦海中想像未來**（內部模擬）
- 🎮 **在虛擬環境中規劃**（零樣本遷移）

---

## 📋 系統架構圖

```
┌─────────────────────────────────────────────────────────┐
│                   物理感知層                              │
│  Variational Encoder: o_t → z_t (壓縮 + 去噪)            │
└─────────────────┬───────────────────────────────────────┘
                  │ z_t (潛在特徵)
┌─────────────────▼───────────────────────────────────────┐
│                   世界模擬層 (RSSM)                       │
│  ┌──────────────────────────────────────────────┐       │
│  │ Prior Path:  h_{t-1}, a_{t-1} → z_t (想像)  │       │
│  │ Posterior Path: h_{t-1}, o_t → z_t (校準)    │       │
│  └──────────────────────────────────────────────┘       │
│            ↓ KL(Prior || Posterior) → 0                 │
│       (當這個趨近0時，AI理解了物理)                       │
└─────────────────┬───────────────────────────────────────┘
                  │ (h_t, z_t) - 內部狀態
┌─────────────────▼───────────────────────────────────────┐
│                   代理控制層                              │
│  Reward Model:  r_t = R(s_t)                            │
│  Value Model:   V(s_t) = E[Σ γ^k r_{t+k}]              │
│  Actor:         π(a|s) - 策略網絡                        │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
# Clone repository
git clone https://github.com/RainesTaiwan/Project-Physics-AGI.git
cd Project-Physics-AGI

# Create conda environment
conda create -n physics-agi python=3.10
conda activate physics-agi

# Install dependencies
pip install -r requirements.txt

# Install MuJoCo (for physics simulation)
pip install mujoco dm-control
```

### 2. 訓練模型

```bash
# Train on DMControl Walker-Walk task
python train.py --config config.yaml --steps 1000000

# Train with custom config
python train.py --config my_config.yaml --steps 500000

# Resume from checkpoint
python train.py --config config.yaml --checkpoint logs/physics_agi_v1/checkpoints/checkpoint_100000.pt
```

### 3. 評估模型

```bash
# Evaluate trained model
python evaluate.py \
    --config config.yaml \
    --checkpoint logs/physics_agi_v1/checkpoints/checkpoint_1000000.pt \
    --episodes 10 \
    --deterministic
```

### 4. 監控訓練

```bash
# Launch TensorBoard
tensorboard --logdir logs/

# Open browser: http://localhost:6006
```

---

## 🎯 核心模組詳解

### 模組 A：變分感知編碼器 (Variational Encoder)

**檔案**: [src/models/encoder.py](src/models/encoder.py)

**功能**：
- 將高維圖像 (64×64×3) 壓縮到低維潛在空間 (32維)
- **信息瓶頸 (Information Bottleneck)**：強制只保留物理必要信息
- 概率性編碼：捕捉測量不確定性

**數學原理**：
```
q(z|o) = N(μ_enc(o), σ_enc(o))
z ~ q(z|o)
```

### 模組 B：循環狀態空間模型 (RSSM)

**檔案**: [src/models/rssm.py](src/models/rssm.py)

**功能**：系統的「物理引擎」

**雙路徑設計**：
1. **Prior (想像路徑)**：`p(z_t | h_t)` - 純憑內部動力學預測
2. **Posterior (真實路徑)**：`q(z_t | h_t, o_t)` - 結合觀測校準

**物理理解指標**：
```
KL(q(z_t|h_t,o_t) || p(z_t|h_t)) → 0
```
當這個值趨近 0 時，代表 AI 能準確預測物理現象

### 模組 C & D：Actor-Critic 系統

**檔案**: [src/models/actor_critic.py](src/models/actor_critic.py)

**組件**：
- **Reward Model**: 預測即時獎勵 `r_t`
- **Value Model**: 評估長期價值 `V(s_t)`
- **Actor**: 輸出動作策略 `π(a|s)`

---

## 📊 訓練流程

### 階段 1：Dynamics Learning (學習物理)

```python
Loss = λ_recon * ||o_t - ô_t||² 
     + λ_kl * KL(Posterior || Prior)
     + λ_reward * ||r_t - r̂_t||²
```

**目標**：
1. 重建觀測 (證明沒有丟失信息)
2. 最小化 KL 散度 (學習物理法則)
3. 預測獎勵 (任務相關)

### 階段 2：Behavior Learning (學習策略)

在 **想像空間 (Latent Space)** 中展開 15 步軌跡：

```python
# Imagine rollout
for t in range(imagination_horizon):
    a_t ~ π(·|s_t)           # Sample action
    s_{t+1} ~ p(·|s_t, a_t)  # Predict next state (dreaming)
    r_t = R(s_t)             # Predict reward
```

**優勢**：
- ✅ 不需要在真實環境中試錯
- ✅ 可以快速規劃（15 步只需 <1ms）
- ✅ 安全探索（不會損壞硬件）

---

## 🔬 關鍵數學約束

### 1. 信息瓶頸 (Information Bottleneck)

```python
latent_dim << input_dim
32 << (64 × 64 × 3) = 12288
壓縮比: 384倍
```

**作用**：迫使 AI 學會「抽象」和「泛化」，而非記憶

### 2. 多步預測一致性 (Long-term Consistency)

```python
# Latent Overshooting
for k in range(overshooting_distance):
    z_{t+k} = RSSM.imagine_step(z_t, a_{t:t+k})
    loss += ||z_{t+k} - z_{t+k}^{real}||²
```

**作用**：確保長期預測不會發散

### 3. KL 平衡 (KL Balancing)

```python
KL_loss = α * KL(Post || Prior) + (1-α) * KL(Prior || Post)
```

**作用**：防止後驗崩塌 (Posterior Collapse)

---

## 📂 項目結構

```
Project-Physics-AGI/
├── config.yaml              # 系統配置
├── requirements.txt         # Python 依賴
├── train.py                 # 訓練腳本
├── evaluate.py              # 評估腳本
├── src/
│   ├── models/
│   │   ├── encoder.py       # 模組 A: 編碼器/解碼器
│   │   ├── rssm.py          # 模組 B: RSSM
│   │   └── actor_critic.py  # 模組 C/D: Actor-Critic
│   ├── trainer.py           # 訓練器
│   └── utils/
│       ├── replay_buffer.py # 經驗回放緩衝區
│       └── env_wrapper.py   # 環境包裝器
└── logs/                    # 訓練日誌和 checkpoints
```

---

## 🎮 支持的環境

### DeepMind Control Suite (推薦)

```yaml
environment:
  name: "DMC-walker-walk"      # Walker: Walk
  # name: "DMC-cheetah-run"    # Cheetah: Run
  # name: "DMC-cartpole-swingup" # CartPole: Swing Up
  backend: "mujoco"
```

### OpenAI Gymnasium

```yaml
environment:
  name: "HalfCheetah-v4"
  # name: "Hopper-v4"
  # name: "Ant-v4"
```

---

## 🔧 配置說明

### 關鍵超參數

```yaml
# Latent dimensions
encoder:
  latent_dim: 32              # 隨機潛在變量維度
rssm:
  deterministic_size: 200     # 確定性記憶維度

# KL divergence
rssm:
  free_nats: 3.0              # 最小 KL 閾值
  kl_balance_scale: 0.8       # KL 平衡係數

# Imagination
rssm:
  imagination_horizon: 15     # 想像未來步數

# Training
training:
  model_lr: 6e-4              # World Model 學習率
  actor_lr: 8e-5              # Actor 學習率
  sequence_length: 50         # 訓練序列長度
  batch_size: 50              # Batch 大小
```

---

## 📈 實驗結果追蹤

### TensorBoard 指標

**Dynamics (物理學習)**：
- `dynamics/reconstruction_loss` - 重建誤差
- `dynamics/kl_loss` - Prior/Posterior KL 散度 (越低 = 越理解物理)
- `dynamics/reward_loss` - 獎勵預測誤差

**Behavior (策略學習)**：
- `behavior/actor_loss` - 策略梯度損失
- `behavior/value_loss` - 價值函數誤差
- `behavior/mean_return` - 想像軌跡的回報

**Collection (數據收集)**：
- `collect/mean_episode_reward` - 真實環境平均獎勵
- `collect/mean_episode_length` - 平均 episode 長度

---

## 🧪 測試模組

每個核心模組都有獨立測試：

```bash
# Test encoder
python -m src.models.encoder

# Test RSSM
python -m src.models.rssm

# Test actor-critic
python -m src.models.actor_critic

# Test replay buffer
python -m src.utils.replay_buffer

# Test environment
python -m src.utils.env_wrapper
```

---

## 🚧 已知限制與未來工作

### 當前限制：
- ⚠️ 僅支持連續動作空間 (離散動作需要修改 Actor)
- ⚠️ 圖像輸入限制在 64×64 (更高分辨率需要更大網絡)
- ⚠️ 單智能體系統 (多智能體需要擴展)

### 未來改進：
- 🔜 支持 Vision Transformer (ViT) 編碼器
- 🔜 整合 Isaac Gym (GPU 加速物理)
- 🔜 實現 Dreamer v3 的改進
- 🔜 多模態輸入 (視覺 + 觸覺 + 本體感知)

---

## 📚 參考文獻

1. **DreamerV2**: Mastering Atari with Discrete World Models (Hafner et al., 2021)
2. **PlaNet**: A Deep Planning Network for Reinforcement Learning (Hafner et al., 2019)
3. **World Models**: Learning and Planning with Latent Dynamics (Ha & Schmidhuber, 2018)

---

## 📄 License

MIT License

---

**⭐ 如果這個項目對你有幫助，請給個 Star！**