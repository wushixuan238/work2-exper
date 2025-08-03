# train_stage1_autoencoder_optical_only.py

import argparse
import json
import os

import torch
import torch.nn as nn
import wandb
from einops import rearrange

from FoMo.model_zoo.multimodal_mae import MultiSpectralViT, Transformer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from FoMo.model_zoo.multimodal_mae import MultiSpectralViT, Transformer
from yujun.dataset.fomo_dataset import FomoCompatibleDataset


# models/fomo_joint_autoencoder.py


class FomoJointAutoencoder(nn.Module):
    """
    一个统一编码器、双解码器的联合自编码器。
    - 编码器: 共享的、预训练的MultiSpectralViT (FoMo-Net)。
    - 解码器: 模态特定的、独立的解码器 (Decoder_sar, Decoder_opt)。
    """

    def __init__(
            self,
            fomo_encoder: MultiSpectralViT,
            decoder_dim=512,
            decoder_depth=6,
            sar_channels=1,
            opt_channels=3,
            image_size=256,
            patch_size=16
    ):
        super().__init__()

        self.encoder = fomo_encoder
        self.patch_size = patch_size
        encoder_dim = self.encoder.transformer.norm.normalized_shape[0]
        num_patches_per_channel = (image_size // patch_size) ** 2

        # 为SAR和OPT创建独立的解码器
        self.decoder_sar = self._create_decoder(
            num_patches=num_patches_per_channel * sar_channels,
            latent_dim=encoder_dim, decoder_dim=decoder_dim, decoder_depth=decoder_depth,
            out_chans=sar_channels, patch_size=patch_size
        )
        self.decoder_opt = self._create_decoder(
            num_patches=num_patches_per_channel * opt_channels,
            latent_dim=encoder_dim, decoder_dim=decoder_dim, decoder_depth=decoder_depth,
            out_chans=opt_channels, patch_size=patch_size
        )

    def _create_decoder(self, num_patches, latent_dim, decoder_dim, decoder_depth, out_chans, patch_size):
        # 这是一个完整的解码器子模块
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc_to_dec = nn.Linear(latent_dim, decoder_dim) if latent_dim != decoder_dim else nn.Identity()
                self.decoder_pos_emb = nn.Parameter(torch.randn(1, num_patches, decoder_dim))
                self.decoder_transformer = Transformer(
                    dim=decoder_dim, depth=decoder_depth, heads=8,
                    dim_head=decoder_dim // 8, mlp_dim=decoder_dim * 4
                )
                self.to_pixels = nn.Linear(decoder_dim, patch_size * patch_size)
                self.patch_size = patch_size
                self.out_chans = out_chans

            def forward(self, tokens, h_patch_count, w_patch_count):
                tokens = self.enc_to_dec(tokens)
                tokens += self.decoder_pos_emb
                decoded_tokens = self.decoder_transformer(tokens)

                # 将tokens重组成每个通道的patches
                tokens_per_channel = torch.chunk(decoded_tokens, self.out_chans, dim=1)

                recon_channels = []
                for chan_tokens in tokens_per_channel:
                    recon_patch = self.to_pixels(chan_tokens)
                    recon_chan_image = rearrange(recon_patch,
                                                 'b (h w) (p1 p2) -> b 1 (h p1) (w p2)',
                                                 h=h_patch_count, w=w_patch_count,
                                                 p1=self.patch_size, p2=self.patch_size)
                    recon_channels.append(recon_chan_image)

                return torch.cat(recon_channels, dim=1)

        return Decoder()

    def forward(self, data, modality):
        images, keys = data
        B, C, H, W = images.shape
        patch_size = self.patch_size

        h_patch_count = H // patch_size
        w_patch_count = W // patch_size

        encoded_tokens = self.encoder(data, pool=False)

        if modality == 'sar':
            reconstructed_image = self.decoder_sar(encoded_tokens, h_patch_count, w_patch_count)
        elif modality == 'opt':
            reconstructed_image = self.decoder_opt(encoded_tokens, h_patch_count, w_patch_count)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        return reconstructed_image, encoded_tokens


# 放在你的训练脚本的开头部分
def custom_collate_fn(batch):
    """
    自定义的collate_fn，用于正确处理 (image, keys_list) 的批次。
    """
    # batch 是一个列表，每个元素是 (image_tensor, keys_list)

    # 1. 分离图像和keys
    images = [item[0] for item in batch]
    keys = [item[1] for item in batch]  # keys现在是 [[9,8,7], [9,8,7], ...]

    # 2. 正常堆叠图像
    images_batch = torch.stack(images, dim=0)

    # 3. 将keys保持为列表的列表，不转换为张量
    # 因为对于一个batch，我们只需要一份keys即可
    keys_batch = keys[0]

    return images_batch, keys_batch


def train_stage1_optical_autoencoder(args):
    device = torch.device(args.device)

    # 1. 初始化wandb
    if args.use_wandb:
        wandb.init(project=args.project_name, name="stage1_optical_autoencoder", config=args)

    # 2. 加载配置文件和预训练编码器
    print(f"Loading config from {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = json.load(f)

        # --- 关键修改点 ---
        # MultiSpectralViT 需要一个列表或元组来确定大小，而不是字典。
        # 我们可以直接使用字典的 values()，或者 keys() 的列表。
        # 为了保证顺序和长度正确，最安全的是基于keys生成一个列表。
        # 确保keys是整数并排序，以创建一个连续的列表。
    modality_keys = sorted([int(k) for k in config['modality_channels'].keys()])
    # +1 是因为字典中最大的key可能不是len-1
    num_total_bands = max(modality_keys) + 1
    # 创建一个简单的列表，其长度足以覆盖所有定义的key
    modality_channels_for_init = list(range(num_total_bands))

    print(f"DEBUG_INIT: config['modality_channels'].keys() = {config['modality_channels'].keys()}")
    print(f"DEBUG_INIT: modality_keys = {modality_keys}")
    print(f"DEBUG_INIT: num_total_bands = {num_total_bands}")
    print(f"DEBUG_INIT: len(modality_channels_for_init) = {len(modality_channels_for_init)}")
    print(f"DEBUG_INIT: modality_channels_for_init = {modality_channels_for_init}")

    # 构建用于初始化的configs字典
    fomo_init_configs = {
        'single_embedding_layer': True,  # 或者从你的config文件中读取
        'modality_channels': modality_channels_for_init
    }

    print("Initializing FoMo-Net encoder...")
    fomo_encoder = MultiSpectralViT(
        image_size=args.image_size, patch_size=args.patch_size, channels=1,
        num_classes=1000, dim=args.encoder_dim, depth=args.encoder_depth,
        heads=args.encoder_heads, mlp_dim=args.encoder_mlp_dim,
        configs=fomo_init_configs
    )

    # print(f"Loading pretrained FoMo-Net weights from: {args.fomo_ckpt_path}")
    # fomo_encoder.load_state_dict(torch.load(args.fomo_ckpt_path, map_location='cpu'), strict=False)

    # 加载预训练权重
    pretrained_state_dict = torch.load(args.fomo_ckpt_path, map_location='cpu')

    # 移除不匹配的 'pos_embedding' 键
    if 'pos_embedding' in pretrained_state_dict:
        print("Ignoring 'pos_embedding' from pretrained checkpoint due to size mismatch.")
        del pretrained_state_dict['pos_embedding']

    # 加载修改后的 state_dict，使用 strict=False 确保其他不匹配（如分类头）被忽略
    fomo_encoder.load_state_dict(pretrained_state_dict, strict=False)

    print("Pretrained FoMo-Net encoder loaded successfully!")

    # 3. 初始化完整的联合自编码器模型
    model = FomoJointAutoencoder(
        fomo_encoder=fomo_encoder,
        decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth,
        sar_channels=args.sar_channels, opt_channels=args.opt_channels,
        image_size=args.image_size, patch_size=args.patch_size
    ).to(device)

    # 4. 数据准备
    print("Preparing SAR and Optical datasets...")
    transform_sar = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    transform_opt = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    sar_dataset = FomoCompatibleDataset(
        data_dir=args.sar_data_dir, dataset_name='my_sar_train_dataset',
        config=config, transform=transform_sar, in_chans=args.sar_channels
    )
    opt_dataset = FomoCompatibleDataset(
        data_dir=args.opt_data_dir, dataset_name='my_optical_train_dataset',
        config=config, transform=transform_opt, in_chans=args.opt_channels
    )

    sar_loader = DataLoader(sar_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=custom_collate_fn)
    opt_loader = DataLoader(opt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=custom_collate_fn)

    print(f"Loaded {len(sar_dataset)} SAR images and {len(opt_dataset)} Optical images.")

    # 5. 优化器和损失函数 (差异化学习率)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': model.decoder_sar.parameters(), 'lr': args.decoder_lr},
        {'params': model.decoder_opt.parameters(), 'lr': args.decoder_lr}
    ], weight_decay=args.weight_decay)

    criterion_recon = nn.L1Loss()
    # if args.use_lpips:
    #     criterion_perceptual = LPIPS(net='alex').to(device)

    # 训练循环
    print("Starting Stage 1: Joint Autoencoder Finetuning...")
    for epoch in range(args.num_epochs):
        model.train()

        sar_iter = iter(sar_loader)
        opt_iter = iter(opt_loader)
        num_batches = min(len(sar_loader), len(opt_loader))

        epoch_loss_sar = 0
        epoch_loss_opt = 0

        with tqdm(range(num_batches), desc=f"Stage 1 - Epoch {epoch + 1}/{args.num_epochs}") as pbar:
            for i in pbar:
                optimizer.zero_grad()

                # --- 联合训练步骤 ---
                # a. SAR分支
                sar_images, sar_keys = next(sar_iter)  # 直接得到一个tensor和一个list
                sar_images = sar_images.to(device)
                sar_data = (sar_images, sar_keys)

                recon_sar_images, _ = model(sar_data, modality='sar')
                loss_sar = criterion_recon(recon_sar_images, sar_images)

                # b. OPTICAL分支
                opt_images, opt_keys = next(opt_iter)
                opt_images = opt_images.to(device)
                opt_data = (opt_images, opt_keys)

                recon_opt_images, _ = model(opt_data, modality='opt')
                loss_opt = criterion_recon(recon_opt_images, opt_images)

                # if args.use_lpips:
                #     loss_opt += args.lambda_lpips * criterion_perceptual(recon_opt_images, opt_images)

                # c. 合并损失并反向传播
                total_loss = loss_sar + loss_opt
                total_loss.backward()
                optimizer.step()

                epoch_loss_sar += loss_sar.item()
                epoch_loss_opt += loss_opt.item()

                pbar.set_postfix({
                    "loss_sar": f"{loss_sar.item():.4f}",
                    "loss_opt": f"{loss_opt.item():.4f}",
                    "total_loss": f"{total_loss.item():.4f}"
                })

                if args.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "step": epoch * num_batches + i,
                        "total_loss_step": total_loss.item(),
                        "loss_sar_step": loss_sar.item(),
                        "loss_opt_step": loss_opt.item(),
                    })

        # 记录epoch平均损失
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "loss_sar_epoch_avg": epoch_loss_sar / num_batches,
                "loss_opt_epoch_avg": epoch_loss_opt / num_batches,
            })

    print("Stage 1 joint training complete!")

    # 保存模型
    output_path = os.path.join(args.output_dir, "fomo_joint_autoencoder_stage1.pt")
    torch.save(model.state_dict(), output_path)
    print(f"Stage 1 model saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 1: Finetune a FoMo-Net based Autoencoder on Optical data ONLY")

    # --- 路径参数 ---
    parser.add_argument("--opt_data_dir", type=str, default='/home/wushixuan/yujun/data/rsdiffusion/sar2opt/trainA',
                        help="Path to Optical data directory")
    parser.add_argument("--sar_data_dir", type=str, default='/home/wushixuan/yujun/data/rsdiffusion/sar2opt/trainB',
                        help="Path to SAR data directory")
    parser.add_argument("--config_path", type=str,
                        default='/home/wushixuan/桌面/07/work2-exper/FoMo/configs/datasets/fomo_pretraining_datasets.json',
                        help="Path to FoMo-Net config file")
    parser.add_argument("--fomo_ckpt_path", type=str,
                        default='/home/wushixuan/yujun/data/weights/fomo_single_embedding_layer_weights.pt',
                        help="Path to pretrained FoMo-Net checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage1_optical",
                        help="Directory to save checkpoints")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)

    parser.add_argument("--sar_channels", type=int, default=1)
    parser.add_argument("--opt_channels", type=int, default=3)

    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--encoder_heads", type=int, default=12)
    parser.add_argument("--encoder_mlp_dim", type=int, default=2048)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_depth", type=int, default=8)
    parser.add_argument("--decoder_heads", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
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
