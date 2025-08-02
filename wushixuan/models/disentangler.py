# models/disentangler.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# --- 梯度反转层 (与你之前的代码相同) ---
class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# --- 解耦器模型 ---
class FeatureDisentangler(nn.Module):
    """
    在固定的潜在空间中，将混合特征解耦为内容和风格。
    输入: 编码器输出的混合特征 tokens (B, N, D)
    输出: unrelated_tokens, related_tokens
    """

    def __init__(self, feature_dim, num_layers=2, hidden_dim_ratio=2):
        super().__init__()
        # 使用简单的MLP作为投影器
        self.unrelated_projector = self._build_mlp(feature_dim, num_layers, hidden_dim_ratio)
        self.related_projector = self._build_mlp(feature_dim, num_layers, hidden_dim_ratio)

    def _build_mlp(self, dim, depth, hidden_ratio):
        layers = [nn.Linear(dim, dim * hidden_ratio), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(dim * hidden_ratio, dim * hidden_ratio), nn.ReLU()])
        layers.append(nn.Linear(dim * hidden_ratio, dim))
        return nn.Sequential(*layers)

    def forward(self, mixed_tokens):
        unrelated_tokens = self.unrelated_projector(mixed_tokens)
        related_tokens = self.related_projector(mixed_tokens)
        return unrelated_tokens, related_tokens


# --- 判别器/混淆器模型 ---
class ModalityDiscriminator(nn.Module):
    """
    判别器，用于判断一个特征向量属于哪个模态。
    输入: 特征 tokens (B, N, D) -> (B*N, D)
    输出: 模态分类 logits (B*N, 2)
    """

    def __init__(self, feature_dim, num_layers=3, hidden_dim=256):
        super().__init__()
        layers = [nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
        layers.append(nn.Linear(hidden_dim, 2))  # 2 for SAR vs. Optical
        self.classifier = nn.Sequential(*layers)

    def forward(self, tokens):
        # (B, N, D) -> (B*N, D)
        b, n, d = tokens.shape
        tokens_flat = tokens.view(b * n, d)
        logits = self.classifier(tokens_flat)
        return logits