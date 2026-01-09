"""
Module A: Variational Sensory Encoder
變分感知編碼器 - 負責將高維傳感器數據降維成緊湊的潛在狀態
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VariationalEncoder(nn.Module):
    """
    變分感知編碼器 (Probabilistic Feature Extraction)
    
    目的：濾除物理世界中的視覺噪聲，提取因果特徵
    
    輸入: o_t (當前幀原始數據，如 RGB 圖像)
    輸出: z_t (隨機潛在變量 Stochastic Latent Variable)
    
    關鍵特性：
    1. 概率性 (Probabilistic) 而非確定性
    2. 信息瓶頸 (Information Bottleneck)
    3. 捕捉測量不確定性
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 64, 64),
        latent_dim: int = 32,
        hidden_dims: list = [32, 64, 128, 256],
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Build convolutional encoder
        channels = [input_shape[0]] + hidden_dims
        layers = []
        
        for i in range(len(hidden_dims)):
            layers.extend([
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU() if activation == "relu" else nn.ELU()
            ])
        
        self.conv_encoder = nn.Sequential(*layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.conv_encoder(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)
        
        # Probabilistic bottleneck: output mean and log_std
        self.fc_mean = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logstd = nn.Linear(self.flattened_size, latent_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder
        
        Args:
            observation: [batch_size, C, H, W] raw sensor data
            
        Returns:
            z: [batch_size, latent_dim] sampled latent state
            mean: [batch_size, latent_dim] predicted mean
            std: [batch_size, latent_dim] predicted standard deviation
        """
        # Normalize input to [-0.5, 0.5]
        x = observation / 255.0 - 0.5
        
        # Convolutional feature extraction
        features = self.conv_encoder(x)
        features = features.view(features.size(0), -1)
        
        # Compute probabilistic latent distribution
        mean = self.fc_mean(features)
        logstd = self.fc_logstd(features)
        
        # Clamp log_std for numerical stability
        logstd = torch.clamp(logstd, -10, 2)
        std = torch.exp(logstd)
        
        # Reparameterization trick: z = μ + σ * ε
        z = self.reparameterize(mean, std)
        
        return z, mean, std
    
    def reparameterize(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling
        z = μ + σ * ε, where ε ~ N(0, 1)
        """
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def encode_deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Deterministic encoding (use mean only, no sampling)
        Used during inference for stability
        """
        x = observation / 255.0 - 0.5
        features = self.conv_encoder(x)
        features = features.view(features.size(0), -1)
        mean = self.fc_mean(features)
        return mean


class CNNDecoder(nn.Module):
    """
    解碼器 - 用於重建觀測，驗證編碼器是否保留了必要信息
    
    輸入: z_t (潛在狀態)
    輸出: ô_t (重建的觀測)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        output_shape: Tuple[int, int, int] = (3, 64, 64),
        hidden_dims: list = [256, 128, 64, 32]
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        
        # Calculate initial spatial size after upsampling
        # For 64x64 output with 4 upsample layers: 4x4 initial size
        self.init_size = output_shape[1] // (2 ** len(hidden_dims))
        self.init_channels = hidden_dims[0]
        
        # Linear projection from latent to feature map
        self.fc = nn.Linear(
            latent_dim,
            self.init_channels * self.init_size * self.init_size
        )
        
        # Transpose convolution layers for upsampling
        layers = []
        channels = hidden_dims + [output_shape[0]]
        
        for i in range(len(hidden_dims)):
            layers.extend([
                nn.ConvTranspose2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(channels[i + 1]) if i < len(hidden_dims) - 1 else nn.Identity(),
                nn.ReLU() if i < len(hidden_dims) - 1 else nn.Identity()
            ])
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to observation
        
        Args:
            z: [batch_size, latent_dim] latent state
            
        Returns:
            reconstruction: [batch_size, C, H, W] reconstructed observation
        """
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        
        # Upsample to original size
        x = self.decoder(x)
        
        # Sigmoid to constrain output to [0, 1]
        x = torch.sigmoid(x)
        
        # Scale back to [0, 255]
        reconstruction = x * 255.0
        
        return reconstruction


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    計算重建損失 (Reconstruction Loss)
    
    使用 MSE 或 Negative Log-Likelihood
    """
    # Normalize both to [0, 1]
    pred_norm = prediction / 255.0
    target_norm = target / 255.0
    
    # MSE Loss
    loss = F.mse_loss(pred_norm, target_norm, reduction=reduction)
    
    return loss


def kl_divergence_gaussian(
    mean: torch.Tensor,
    std: torch.Tensor,
    free_nats: float = 3.0
) -> torch.Tensor:
    """
    計算高斯分佈的 KL 散度
    KL(q(z|x) || p(z)) where p(z) = N(0, I)
    
    Args:
        mean: [batch_size, latent_dim]
        std: [batch_size, latent_dim]
        free_nats: minimum KL (prevents collapse to deterministic)
        
    Returns:
        kl_loss: scalar
    """
    # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (μ² + σ² - log(σ²) - 1)
    var = std ** 2
    kl = 0.5 * (mean ** 2 + var - torch.log(var + 1e-8) - 1)
    
    # Sum over latent dimensions
    kl = kl.sum(dim=-1)
    
    # Apply free nats threshold
    kl = torch.maximum(kl, torch.tensor(free_nats).to(kl.device))
    
    # Average over batch
    return kl.mean()


if __name__ == "__main__":
    # Test encoder
    print("Testing Variational Encoder...")
    encoder = VariationalEncoder(
        input_shape=(3, 64, 64),
        latent_dim=32,
        hidden_dims=[32, 64, 128, 256]
    )
    
    # Test with random image
    batch_size = 4
    dummy_obs = torch.randint(0, 256, (batch_size, 3, 64, 64)).float()
    
    z, mean, std = encoder(dummy_obs)
    print(f"Observation shape: {dummy_obs.shape}")
    print(f"Latent z shape: {z.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Compression ratio: {dummy_obs.numel() / z.numel():.2f}x")
    
    # Test decoder
    print("\nTesting CNN Decoder...")
    decoder = CNNDecoder(latent_dim=32, output_shape=(3, 64, 64))
    
    reconstruction = decoder(z)
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Test reconstruction loss
    recon_loss = reconstruction_loss(reconstruction, dummy_obs)
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    
    # Test KL divergence
    kl_loss = kl_divergence_gaussian(mean, std)
    print(f"KL divergence: {kl_loss.item():.4f}")
    
    print("\n✓ Module A: Variational Encoder Test Passed!")
