# models/fomo_shared_autoencoder.py

import torch
import torch.nn as nn
from FoMo.model_zoo.multimodal_mae import MultiSpectralViT


class FomoSharedAutoencoder(nn.Module):
    """
    使用一个统一的MultiSpectralViT作为共享编码器，
    并为SAR和Optical模态配备独立的解码器。
    """

    def __init__(
            self,
            fomo_encoder: MultiSpectralViT,  # 传入已经实例化的FoMo-Net
            decoder_dim=512,
            decoder_depth=6,
            sar_channels=2,
            opt_channels=3
    ):
        super().__init__()

        # 1. 共享编码器 (Shared Encoder)
        # 整个FoMo-Net (MultiSpectralViT) 都作为共享编码器
        self.encoder = fomo_encoder
        encoder_dim = self.encoder.transformer.norm.normalized_shape[0]

        # 2. 模态特定的解码器 (Specific Decoders)
        # 解码器接收编码器输出的、拼接了所有通道的token序列
        # 注意：这里需要知道每个模态有多少个patch
        num_patches_per_channel = self.encoder.num_patches

        self.decoder_sar = self.create_decoder(
            num_patches=num_patches_per_channel * sar_channels,
            latent_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            out_chans=sar_channels,
            patch_size=self.encoder.to_patch_embedding.patch_height  # 从encoder获取patch_size
        )

        self.decoder_opt = self.create_decoder(
            num_patches=num_patches_per_channel * opt_channels,
            latent_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            out_chans=opt_channels,
            patch_size=self.encoder.to_patch_embedding.patch_height
        )

    def create_decoder(self, num_patches, latent_dim, decoder_dim, decoder_depth, out_chans, patch_size):
        # 这是一个更完整的解码器示例，借鉴MAE
        from FoMo.model_zoo.multimodal_mae import Transformer
        from einops import rearrange

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc_to_dec = nn.Linear(latent_dim, decoder_dim) if latent_dim != decoder_dim else nn.Identity()
                self.decoder_pos_emb = nn.Parameter(torch.randn(1, num_patches, decoder_dim))
                self.decoder_transformer = Transformer(
                    dim=decoder_dim, depth=decoder_depth, heads=8,
                    dim_head=decoder_dim // 8, mlp_dim=decoder_dim * 4
                )
                self.to_pixels = nn.Linear(decoder_dim, patch_size * patch_size * out_chans)
                self.patch_size = patch_size
                self.out_chans = out_chans

            def forward(self, tokens):
                tokens = self.enc_to_dec(tokens)
                tokens += self.decoder_pos_emb
                decoded_tokens = self.decoder_transformer(tokens)

                recon_patches = self.to_pixels(decoded_tokens)

                # 重组回图像
                # 注意：这里的h, w计算需要小心
                h_w_ratio = 1  # 假设是方形patch
                num_patches_per_channel = num_patches // self.out_chans
                h_patch_count = int(num_patches_per_channel ** 0.5)

                recon_image = rearrange(recon_patches,
                                        'b (c h w) (p1 p2) -> b c (h p1) (w p2)',
                                        c=self.out_chans, h=h_patch_count, w=h_patch_count,
                                        p1=self.patch_size, p2=self.patch_size)
                return recon_image

        return Decoder()

    def forward(self, data, modality):
        # 1. 使用共享编码器编码
        # pool=False得到所有patch的tokens
        encoded_tokens = self.encoder(data, pool=False)

        # 2. 根据模态选择对应的解码器
        if modality == 'sar':
            reconstructed_image = self.decoder_sar(encoded_tokens)
        elif modality == 'opt':
            reconstructed_image = self.decoder_opt(encoded_tokens)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # 注意：这里的shared_features就是encoded_tokens本身，因为整个encoder是共享的
        return reconstructed_image, encoded_tokens
