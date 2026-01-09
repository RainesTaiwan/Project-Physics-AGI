# 🎓 Project Physics-AGI - 技術文檔

## 📖 目錄

1. [系統概述](#系統概述)
2. [核心概念](#核心概念)
3. [架構深度解析](#架構深度解析)
4. [訓練策略](#訓練策略)
5. [數學推導](#數學推導)
6. [實驗指南](#實驗指南)

---

## 系統概述

### 什麼是 World Model？

World Model 是一個 **內部模擬器**，能夠：

1. **理解物理** - 學習環境的動力學方程
2. **預測未來** - 在潛在空間中「想像」未來狀態
3. **安全規劃** - 在虛擬環境中試錯，而非真實世界

### 與傳統 RL 的區別

| 特性 | 傳統 RL (Model-Free) | World Model (Model-Based) |
|------|---------------------|--------------------------|
| 學習對象 | 直接學習策略 π(a\|s) | 學習動力學 p(s'\|s,a) + 策略 |
| 樣本效率 | 低（需要大量交互） | 高（可在想像中訓練） |
| 泛化能力 | 弱（過擬合特定場景） | 強（理解物理法則） |
| 計算成本 | 低 | 高（需要訓練模型） |

---

## 核心概念

### 1. 狀態分解 (State Decomposition)

World Model 將狀態分解為兩部分：

```
完整狀態 s_t = (h_t, z_t)

h_t: 確定性記憶 (Deterministic State)
     - 來自 RNN/GRU
     - 保存歷史信息
     - 維度: 200

z_t: 隨機潛在變量 (Stochastic Latent)
     - 來自觀測編碼
     - 捕捉不確定性
     - 維度: 32
```

**為什麼這樣設計？**
- `h_t` 負責記憶（如位置、速度的時序依賴）
- `z_t` 負責感知（如當前圖像的視覺特徵）
- 兩者結合 = 完整的世界理解

### 2. 雙路徑學習 (Dual Path Learning)

```
Prior Path (想像路徑):
    p(z_t | h_t) - 只用歷史，預測未來

Posterior Path (真實路徑):
    q(z_t | h_t, o_t) - 結合觀測，校準狀態
```

**訓練目標**：
```
最小化 KL(Posterior || Prior)
```

當 KL → 0 時，意味著：
- Prior 能準確預測（不需要觀測）
- AI 理解了物理法則
- 可以純粹在想像中規劃

### 3. 潛在過沖 (Latent Overshooting)

標準訓練只預測下一步：
```
s_t → s_{t+1}
```

Overshooting 預測多步：
```
s_t → s_{t+1} → s_{t+2} → ... → s_{t+k}
```

**作用**：
- 確保長期預測一致性
- 防止誤差累積
- 提升泛化能力

---

## 架構深度解析

### 模組 A: Variational Encoder

**輸入處理**：
```python
# 1. 歸一化到 [-0.5, 0.5]
x = observation / 255.0 - 0.5

# 2. CNN 特徵提取
features = Conv2D(x)  # [B, C, H, W] → [B, D_flat]

# 3. 概率性瓶頸
μ = Linear(features)      # Mean
log_σ = Linear(features)  # Log std
σ = exp(log_σ)

# 4. 重參數化採樣
ε ~ N(0, 1)
z = μ + σ * ε
```

**信息瓶頸**：
```
Input:  64 × 64 × 3 = 12,288 維
Latent: 32 維
壓縮比: 384倍
```

這迫使模型只能保留「物理必要」的信息，自動過濾噪聲。

### 模組 B: RSSM

**時間展開 (Temporal Rollout)**：

```
時刻 t-1:
    狀態 = (h_{t-1}, z_{t-1})
    動作 = a_{t-1}

時刻 t:
    # Step 1: 更新確定性記憶
    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    
    # Step 2: 預測先驗分佈
    p(z_t | h_t) = N(μ_prior(h_t), σ_prior(h_t))
    
    # Step 3 (訓練時): 結合觀測
    q(z_t | h_t, o_t) = N(μ_post(h_t, o_t), σ_post(h_t, o_t))
    
    # Step 4: 採樣狀態
    z_t ~ q(z_t | h_t, o_t)  # 訓練時用 Posterior
    z_t ~ p(z_t | h_t)       # 推理時用 Prior
```

**KL 平衡技巧**：

標準 KL:
```
KL_forward = KL(q(z|h,o) || p(z|h))
```

問題：可能導致 Posterior Collapse（後驗退化到先驗）

解決方案 - KL Balancing:
```
KL_balanced = α * KL(q || p) + (1-α) * KL(p || q)
            = α * KL_forward + (1-α) * KL_reverse
```

α = 0.8 時效果最好。

### 模組 C/D: Actor-Critic

**Actor (策略)**：
```
輸入: (h_t, z_t)
輸出: Tanh Normal 分佈

π(a | s) = TanhNormal(μ_π(s), σ_π(s))

其中:
    u ~ N(μ, σ)      # Latent action
    a = tanh(u)      # Squash to [-1, 1]
```

**Value (價值)**：
```
V(s_t) = E[Σ_{k=0}^∞ γ^k r_{t+k} | s_t]

訓練目標:
    minimize ||V(s) - R_λ(s)||²
    
其中 R_λ 是 λ-return (GAE)
```

**Reward (獎勵預測)**：
```
r̂_t = R(h_t, z_t)

訓練目標:
    minimize ||r̂_t - r_t||²
```

---

## 訓練策略

### 階段 1: Dynamics Learning

**目標函數**：
```
L_dynamics = L_recon + β_kl * L_kl + β_r * L_reward

其中:
    L_recon = ||o_t - ô_t||²        # 重建損失
    L_kl = KL(q(z|h,o) || p(z|h))   # 動力學損失
    L_reward = ||r_t - r̂_t||²       # 獎勵損失
```

**訓練流程**：
```python
for batch in dataloader:
    # 1. 初始化 RSSM 狀態
    state = rssm.initial_state()
    
    # 2. 序列展開
    for t in range(sequence_length):
        # 編碼觀測
        z_t = encoder(o_t)
        
        # RSSM 更新
        state, prior, posterior = rssm.observe_step(
            state, a_{t-1}, z_t
        )
        
        # 計算損失
        loss_recon += ||decoder(state) - o_t||²
        loss_kl += KL(posterior || prior)
        loss_reward += ||reward_model(state) - r_t||²
    
    # 3. 反向傳播
    total_loss.backward()
    optimizer.step()
```

### 階段 2: Behavior Learning

**在想像中訓練**：
```python
# 1. 從真實數據初始化
state = build_initial_state(real_trajectory)

# 2. 想像未來軌跡
imagined_states = []
imagined_rewards = []

for t in range(imagination_horizon):
    # 從策略採樣動作
    a_t ~ π(·| state)
    
    # 想像下一狀態（純粹用 Prior）
    state = rssm.imagine_step(state, a_t)
    
    # 預測獎勵
    r_t = reward_model(state)
    
    imagined_states.append(state)
    imagined_rewards.append(r_t)

# 3. 計算回報
returns = compute_lambda_returns(imagined_rewards)

# 4. 更新策略
actor_loss = -log π(a|s) * advantage
value_loss = ||V(s) - returns||²
```

**優勢**：
- ✅ 不需要與環境交互（快）
- ✅ 可以並行多條想像軌跡
- ✅ 安全（不會損壞硬件）

---

## 數學推導

### KL 散度推導

對於兩個高斯分佈：
```
p = N(μ_p, σ_p²)
q = N(μ_q, σ_q²)

KL(q || p) = log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²)/(2σ_p²) - 1/2
```

當 p = N(0, 1) 時（標準正態）：
```
KL(q || N(0,1)) = (μ² + σ² - log(σ²) - 1) / 2
```

### λ-Return 推導

```
R_t^λ = r_t + γ[(1-λ)V(s_{t+1}) + λR_{t+1}^λ]

展開:
R_t^λ = Σ_{k=0}^{H-1} (γλ)^k δ_{t+k} + (γλ)^H V(s_{t+H})

其中 δ_t = r_t + γV(s_{t+1}) - V(s_t) 是 TD-error
```

### Tanh Normal 對數概率

```
a = tanh(u), u ~ N(μ, σ)

log π(a) = log p_u(u) - Σ log(1 - a²)
         = log N(u; μ, σ) - Σ log(1 - tanh²(u))

其中第二項是 Jacobian 修正
```

---

## 實驗指南

### 超參數調優

**最關鍵的參數**：

1. **KL Weight (β_kl)**
   - 太高：Prior 不準確，無法想像
   - 太低：Posterior Collapse
   - 推薦：1.0，配合 free_nats=3.0

2. **Imagination Horizon**
   - 太短：策略短視
   - 太長：累積誤差
   - 推薦：15 步

3. **Sequence Length**
   - 影響時序學習
   - 推薦：50 步

### 診斷指標

**物理理解程度**：
```
KL Divergence → 0  ✅ 理解物理
KL Divergence > 10 ⚠️  仍在探索
```

**重建品質**：
```
Reconstruction Loss < 0.1  ✅ 保留信息
Reconstruction Loss > 0.5  ⚠️  信息丟失
```

**策略品質**：
```
Episode Reward ↑  ✅ 策略改進
Episode Reward 平穩 → 收斂
```

### 常見問題

**Q: KL 始終很高？**
A: 
1. 檢查 free_nats 是否太大
2. 增加 kl_weight
3. 確保 RSSM 有足夠容量

**Q: 重建模糊？**
A:
1. 增加解碼器容量
2. 檢查 latent_dim 是否太小
3. 調整 reconstruction_weight

**Q: 策略不改進？**
A:
1. 檢查 reward_model 是否準確
2. 增加 imagination_horizon
3. 調整 actor_lr

---

## 📚 延伸閱讀

1. **DreamerV2 論文**: https://arxiv.org/abs/2010.02193
2. **PlaNet 論文**: https://arxiv.org/abs/1811.04551
3. **World Models 論文**: https://arxiv.org/abs/1803.10122

---

**本文檔持續更新中...**
