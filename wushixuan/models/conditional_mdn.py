# models/conditional_mdn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 借鉴自 rcg.rdm，用于生成时间步嵌入
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    生成 sinusoidal timestep embedding。
    :param timesteps: (N,) shape of timesteps.
    :param dim: embedding dimension.
    :param max_period: The maximum period for the sinusoidal oscillation.
    :return: (N, dim) shape of embeddings.
    """
    if max_period is None:
        max_period = 10000

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ConditionalMDN(nn.Module):
    """
    一个条件化的 MLP 扩散模型。
    用于在给定内容特征的条件下，预测带噪声风格特征的噪声。
    """

    def __init__(
            self,
            feature_dim,
            condition_dim,
            num_layers=8,
            hidden_dim_ratio=4,
            time_embed_dim=256,
            dropout=0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = feature_dim * hidden_dim_ratio
        self.time_embed_dim = time_embed_dim
        self.condition_dim = condition_dim

        # 1. 时间步嵌入（类似 SimpleMLP 的 time_embed）
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # 2. 条件嵌入（类似 SimpleMLP 的 context_layers）
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # 3. 主干网络：使用 ResBlock 风格的 MLP 块
        self.input_proj = nn.Linear(feature_dim, self.hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            ]))

        self.output_proj = nn.Linear(self.hidden_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, noisy_style_tokens, time_steps, content_condition_tokens):
        """
        前向传播。
        :param noisy_style_tokens: 带噪声的风格特征 (B, N, D_feat)
        :param time_steps: 时间步 (B,)
        :param content_condition_tokens: 内容条件特征 (B, N, D_cond)
        """
        # 修正：在解包之前检查张量的维度
        if noisy_style_tokens.dim() == 2:
            noisy_style_tokens = noisy_style_tokens.unsqueeze(1)

        b, n, d = noisy_style_tokens.shape

        # 同样，为 content_condition_tokens 增加维度
        if content_condition_tokens.dim() == 2:
            content_condition_tokens = content_condition_tokens.unsqueeze(1)

        # 将输入展平以便MLP处理
        x = noisy_style_tokens.view(b * n, d)
        cond = content_condition_tokens.view(b * n, -1)

        # 1. 生成时间步嵌入
        time_emb_base = timestep_embedding(time_steps, self.time_embed_dim)
        time_emb = self.time_mlp(time_emb_base)  # (B, hidden_dim)

        # 2. 生成条件嵌入
        # 我们对每个token的条件进行嵌入，而不是对整个序列求平均
        cond_emb = self.condition_mlp(cond)  # (B*N, hidden_dim)

        # 3. 将时间和条件嵌入进行广播，以便与主干网络相加
        time_emb_broadcast = time_emb.unsqueeze(1).repeat(1, n, 1).view(b * n, -1)

        # 4. 通过主干网络
        h = self.input_proj(x)
        h = self.dropout(h)

        for norm, silu, linear in self.layers:
            h = norm(h)
            h = silu(h)
            h = linear(h)

            # 将时间和条件嵌入加到 ResBlock 风格的残差连接中
            h = h + time_emb_broadcast + cond_emb
            h = self.dropout(h)

        x = self.output_proj(h)

        return x.view(b, n, d)
