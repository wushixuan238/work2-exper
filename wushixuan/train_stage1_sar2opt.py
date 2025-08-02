# train_stage1_autoencoder_optical_only.py

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
from pathlib import Path



# from models.autoencoder import FomoAutoencoder  # 模型定义基本不变
# from multimodal_mae import MultiSpectralViT, Transformer
# from utils.config import load_fomo_configs
# from dataset.sar_opt_dataset import SAROptDatasetWithKeys # 数据集类
# from lpips import LPIPS # pip install lpips


class FomoAutoencoder(nn.Module):
    def __init__(self, fomo_encoder, decoder_dim, decoder_depth, decoder_heads, patch_size, image_size, num_channels=3):
        super().__init__()
        self.encoder = fomo_encoder
        encoder_dim = self.encoder.transformer.norm.normalized_shape[0]
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        # from multimodal_mae import Transformer
        # self.decoder_transformer = Transformer(...)
        num_patches_per_channel = (image_size // patch_size) ** 2
        total_patches = num_patches_per_channel * num_channels
        self.decoder_pos_emb = nn.Parameter(torch.randn(1, total_patches, decoder_dim))

        patch_dim = 3 * patch_size * patch_size  # RGB patch
        # self.to_pixels = nn.Linear(decoder_dim, patch_dim)


    def forward(self, data):

        images, _ = data
        reconstructed_image = torch.randn_like(images)
        encoded_tokens = torch.randn(images.shape[0], 1024, 768)  # 假设的token形状
        return reconstructed_image, encoded_tokens


def train_stage1_optical_autoencoder(args):
    """
    第一阶段：只在光学图像上微调/训练一个强大的自编码器。
    """
    device = torch.device(args.device)

    # 1. 初始化wandb
    if args.use_wandb:
        wandb.init(project=args.project_name, name="stage1_optical_autoencoder", config=args)

    # 2. 加载FoMo-Net的配置和预训练编码器
    print("Initializing FoMo-Net encoder...")
    # ... (与之前版本相同，加载预训练的fomo_encoder) ...
    configs = ...
    fomo_encoder = MultiSpectralViT(...)
    fomo_encoder.load_state_dict(...)


    model = FomoAutoencoder(
        fomo_encoder=fomo_encoder,
        decoder_dim=args.decoder_dim,
        ...
        num_channels=3 # 明确告知模型处理3通道
    ).to(device)

    # 4. 数据准备 (核心改动)
    print("Preparing Optical dataset...")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 定义光学图像的光谱ID (R, G, B)
    # 你需要根据FoMo-Net的预训练配置来确定这些ID
    # 假设它们是 [2, 3, 4]
    # optical_keys = [2, 3, 4]

    # !! 只加载光学数据集 !!
    opt_dataset = SAROptDatasetWithKeys(args.opt_data_dir, keys=optical_keys, transform=transform)
    dataloader = DataLoader(
        opt_dataset, # 不再使用ConcatDataset
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Total optical images for training: {len(opt_dataset)}")

    # 5. 优化器和损失函数 (与之前版本相同)
    print("Setting up optimizer and loss functions...")
    # optimizer = torch.optim.AdamW([
    #     {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
    #     {'params': model.decoder.parameters(), 'lr': args.decoder_lr}
    # ], ...)

    # criterion_recon = nn.L1Loss()
    # if args.use_lpips:
    #     criterion_perceptual = LPIPS(net='alex').to(device)

    # 训练循环
    print("Starting Stage 1: Optical-only Autoencoder finetuning...")
    for epoch in range(args.num_epochs):
    model.train()

    for data_batch in tqdm(dataloader, desc=f"Stage 1 - Epoch {epoch+1}/{args.num_epochs}"):
        # !! 数据加载器现在只返回光学数据 !!
        images, keys_list = data_batch
        images = images.to(device)

        # keys_list 现在对于batch里的每个样本都是相同的 [2, 3, 4]
        # FoMo-Net的输入格式仍然是 (tensor, flat_keys_list)
        # 我们需要为batch中的每个样本的每个通道构建keys
        # 例如，如果batch_size=2, keys_list会是 [[2,3,4], [2,3,4]]
        # 我们需要的是 [2, 3, 4, 2, 3, 4, ...] 这样的平铺列表
        # 注意：MultiSpectralViT的实现可能需要调整以适应这种批处理
        # 一个简单的处理方式是假设每个样本的keys是固定的

        optimizer.zero_grad()

        reconstructed, _ = model((images, optical_keys)) # keys是固定的

        # 计算损失 (现在很简单，因为都是3通道RGB)
        recon_loss = criterion_recon(reconstructed, images)
        total_loss = recon_loss

        if args.use_lpips:
            perceptual_loss = criterion_perceptual(reconstructed, images)
            total_loss = total_loss + args.lambda_lpips * perceptual_loss

        total_loss.backward()
        optimizer.step()

        if args.use_wandb:
    #         # wandb 日志记录
    #         ...

    print("Stage 1 (Optical-only) finetuning complete!")

    # 保存最终模型
    output_path = os.path.join(args.output_dir, "fomo_autoencoder_optical_finetuned.pt")
    torch.save(model.state_dict(), output_path)
    print(f"Finetuned optical autoencoder saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 1: Finetune a FoMo-Net based Autoencoder on Optical data ONLY")

    # --- 路径参数 ---
    # !! 只需要光学数据路径 !!
    parser.add_argument("--opt_data_dir", type=str, required=True, help="Path to Optical data directory")
    parser.add_argument("--fomo_config_path", type=str, required=True, help="Path to FoMo-Net config file")
    parser.add_argument("--fomo_ckpt_path", type=str, required=True, help="Path to pretrained FoMo-Net checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage1_optical",
                        help="Directory to save checkpoints")

    # ... (其他参数如 image_size, patch_size, 模型参数, 训练参数, 日志参数等与上一版保持一致) ...
    # 你可以删除 --sar_data_dir 参数，因为它不再需要了
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--encoder_heads", type=int, default=12)
    parser.add_argument("--encoder_mlp_dim", type=int, default=3072)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_depth", type=int, default=8)
    parser.add_argument("--decoder_heads", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--decoder_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--use_lpips", action='store_true')
    parser.add_argument("--lambda_lpips", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--project_name", type=str, default="SAR-to-Optical-Translation")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_stage1_optical_autoencoder(args)