# models/conditional_mdn.py
import torch
import torch.nn as nn


class ConditionalMDN(nn.Module):
    """
    一个条件化的扩散模型，用于在给定内容特征的条件下，生成风格特征。
    架构基于简单的全连接网络（MLP）。
    """

    def __init__(self, feature_dim, condition_dim, num_layers=8, hidden_dim_ratio=4, time_embed_dim=256):
        super().__init__()

        self.feature_dim = feature_dim

        # 1. 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )

        # 2. 条件嵌入
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )

        # 3. 主干网络
        hidden_dim = feature_dim * hidden_dim_ratio

        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ]))

        self.output_proj = nn.Linear(hidden_dim, feature_dim)

        # 4. 用于将时间和条件信息注入到每个块的线性层
        self.time_cond_proj = nn.Linear(time_embed_dim, hidden_dim)

    def forward(self, noisy_style_tokens, time_steps, content_condition_tokens):
        """
        前向传播。

        :param noisy_style_tokens: 带噪声的风格特征 (B, N, D_feat)
        :param time_steps: 时间步 (B,)
        :param content_condition_tokens: 内容条件特征 (B, N, D_cond)
        """
        # (B, N, D) -> (B*N, D)
        b, n, d = noisy_style_tokens.shape

        # 1. 将输入展平以便MLP处理
        x = noisy_style_tokens.view(b * n, d)

        # 2. 嵌入时间和条件
        time_emb = self.time_mlp(time_steps)  # (B, D_time)
        cond_emb = torch.mean(content_condition_tokens, dim=1)  # (B, N, D_cond) -> (B, D_cond)
        cond_emb = self.condition_mlp(cond_emb)  # (B, D_time)

        # 合并嵌入并扩展
        time_cond_emb = self.time_cond_proj(time_emb + cond_emb)  # (B, D_hidden)
        time_cond_emb = time_cond_emb.unsqueeze(1).repeat(1, n, 1).view(b * n, -1)  # (B*N, D_hidden)

        # 3. 通过主干网络
        x = self.input_proj(x)

        for norm, silu, linear in self.layers:
            # 注入时间和条件信息
            h = x + time_cond_emb
            h = norm(h)
            h = silu(h)
            h = linear(h)
            x = x + h  # 残差连接

        x = self.output_proj(x)

        # (B*N, D) -> (B, N, D)
        return x.view(b, n, d)


# --- 需要一个函数来生成时间步嵌入 ---
# 这个可以从扩散模型的标准库中获取，或者自己实现一个
def get_timestep_embedding(timesteps, embedding_dim):
    # Sinusoidal embedding logic
    # ...
    pass