# train_stage2_disentangler.py

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from wushixuan.models.fomo_shared_autoencoder import FomoJointAutoencoder
from FoMo.model_zoo.multimodal_mae import MultiSpectralViT
from models.disentangler import FeatureDisentangler, ModalityDiscriminator, GradientReverseLayer
from wushixuan.train_stage1_sar2opt import custom_collate_fn
from yujun.dataset.fomo_dataset import FomoCompatibleDataset


# import wandb


def train_stage2_disentangler(args):
    device = torch.device(args.device)

    # 1. 初始化wandb
    if args.use_wandb:
        wandb.init(project=args.project_name, name="stage2_disentangler", config=args)

    # --- 1. 加载并冻结第一阶段的自编码器 ---
    print("Loading and freezing the pretrained Stage 1 autoencoder...")

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    modality_keys = sorted([int(k) for k in config['modality_channels'].keys()])
    num_total_bands = max(modality_keys) + 1
    modality_channels_for_init = list(range(num_total_bands))
    fomo_init_configs = {'single_embedding_layer': True, 'modality_channels': modality_channels_for_init}

    fomo_encoder = MultiSpectralViT(
        image_size=args.image_size, patch_size=args.patch_size, channels=1, num_classes=1000,
        dim=args.encoder_dim, depth=args.encoder_depth, heads=args.encoder_heads, mlp_dim=args.encoder_mlp_dim,
        configs=fomo_init_configs
    )

    autoencoder = FomoJointAutoencoder(
        fomo_encoder=fomo_encoder, decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth,
        sar_channels=args.sar_channels, opt_channels=args.opt_channels,
        image_size=args.image_size, patch_size=args.patch_size
    )

    # b. 加载第一阶段训练好的权重
    print(f"Loading Stage 1 checkpoint from: {args.stage1_ckpt_path}")
    autoencoder.load_state_dict(torch.load(args.stage1_ckpt_path, map_location='cpu'))

    # c. 冻结所有参数
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.to(device)
    print("Stage 1 autoencoder is loaded and frozen.")

    # --- 3. 初始化需要训练的新模型 ---
    print("Initializing Stage 2 models (Disentangler, Discriminator, Confuser)...")
    disentangler = FeatureDisentangler(feature_dim=args.encoder_dim).to(device)
    discriminator_related = ModalityDiscriminator(feature_dim=args.encoder_dim).to(device)
    confuser_unrelated = ModalityDiscriminator(feature_dim=args.encoder_dim).to(device)

    # --- 3. 数据准备 (需要模态标签) ---
    print("Preparing datasets with modality labels...")
    # ... (与第一阶段类似, 但需要给Dataset传入modality_label) ...
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
        config=config, transform=transform_sar, in_chans=args.sar_channels, modality_label=0  # SAR标签为0
    )
    opt_dataset = FomoCompatibleDataset(
        data_dir=args.opt_data_dir, dataset_name='my_optical_train_dataset',
        config=config, transform=transform_opt, in_chans=args.opt_channels, modality_label=1  # OPT标签为1
    )
    # 注意：为了让两个数据集一起工作，我们需要一个ConcatDataset
    combined_dataset = ConcatDataset([sar_dataset, opt_dataset])
    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=custom_collate_fn)

    sar_loader = DataLoader(sar_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=custom_collate_fn)
    opt_loader = DataLoader(opt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=custom_collate_fn)

    # --- 5. 优化器和损失函数 ---
    optimizer_disentangler = torch.optim.AdamW(disentangler.parameters(), lr=args.lr_disentangler)
    optimizer_discriminator = torch.optim.AdamW(discriminator_related.parameters(), lr=args.lr_discriminator)
    optimizer_confuser = torch.optim.AdamW(confuser_unrelated.parameters(), lr=args.lr_discriminator)
    criterion_cls = nn.CrossEntropyLoss()

    print("Starting Stage 2: Feature Disentanglement Training...")
    for epoch in range(args.num_epochs):
        disentangler.train()
        discriminator_related.train()
        confuser_unrelated.train()

        with tqdm(dataloader, desc=f"Stage 2 - Epoch {epoch + 1}/{args.num_epochs}") as pbar:
            for images, keys, modality_labels_batch in pbar:
                images = images.to(device)
                modality_labels_batch = modality_labels_batch.to(device)

                # --- 提取冻结的混合特征 ---
                with torch.no_grad():
                    # 我们需要根据模态标签分离图像，分别送入编码器
                    is_sar = (modality_labels_batch == 0)
                    is_opt = (modality_labels_batch == 1)

                    mixed_tokens_list = []
                    if is_sar.any():
                        _, tokens_sar = autoencoder((images[is_sar], keys), modality='sar')
                        mixed_tokens_list.append(tokens_sar)
                    if is_opt.any():
                        _, tokens_opt = autoencoder((images[is_opt], keys), modality='opt')
                        mixed_tokens_list.append(tokens_opt)

                    mixed_tokens = torch.cat(mixed_tokens_list, dim=0)

                # --- 创建用于分类的模态标签 ---
                b, n, d = mixed_tokens.shape
                # 标签需要与token数量对齐 (B*N)
                modality_labels_flat = modality_labels_batch.view(-1, 1).repeat(1, n).view(-1)

                # ============================================
                #  (1) 训练解耦器 (Disentangler)
                # ============================================
                optimizer_disentangler.zero_grad()

                unrelated_tokens, related_tokens = disentangler(mixed_tokens)

                # a. 判别损失
                pred_related = discriminator_related(related_tokens)
                loss_disc = criterion_cls(pred_related, modality_labels_flat)

                # b. 混淆损失
                reversed_unrelated = GradientReverseLayer.apply(unrelated_tokens, args.grl_lambda)
                pred_unrelated = confuser_unrelated(reversed_unrelated)
                loss_conf = criterion_cls(pred_unrelated, modality_labels_flat)

                total_loss_disentangler = args.lambda_disc * loss_disc + args.lambda_conf * loss_conf
                total_loss_disentangler.backward()
                optimizer_disentangler.step()

                # ============================================
                #  (2) 训练判别器 (Discriminator for related features)
                # ============================================
                optimizer_discriminator.zero_grad()

                _, related_tokens_detached = disentangler(mixed_tokens.detach())
                pred_related_for_disc = discriminator_related(related_tokens_detached.detach())
                loss_disc_only = criterion_cls(pred_related_for_disc, modality_labels_flat)
                loss_disc_only.backward()
                optimizer_discriminator.step()

                # ============================================
                #  (3) 训练混淆器 (Confuser for unrelated features)
                # ============================================
                optimizer_confuser.zero_grad()

                unrelated_tokens_detached, _ = disentangler(mixed_tokens.detach())
                pred_unrelated_for_conf = confuser_unrelated(unrelated_tokens_detached.detach())
                loss_conf_only = criterion_cls(pred_unrelated_for_conf, modality_labels_flat)
                loss_conf_only.backward()
                optimizer_confuser.step()

                # ... (日志记录) ...
                pbar.set_postfix({
                    "loss_disentangler": f"{total_loss_disentangler.item():.4f}",
                    "loss_disc": f"{loss_disc_only.item():.4f}",
                    "loss_conf": f"{loss_conf_only.item():.4f}"
                })

    print("Stage 2 feature disentanglement training complete!")

    # 保存模型
    output_path = os.path.join(args.output_dir, "disentangler_stage2.pt")
    torch.save(disentangler.state_dict(), output_path)
    print(f"Stage 2 disentangler saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 2: Train a Feature Disentangler")

    # --- 路径参数 ---
    parser.add_argument("--stage1_ckpt_path", type=str, required=True,
                        help="Path to the trained Stage 1 autoencoder checkpoint")
    parser.add_argument("--sar_data_dir", type=str, required=True)
    parser.add_argument("--opt_data_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage2_disentangler")

    # --- 模型参数 (与第一阶段的编码器匹配) ---
    parser.add_argument("--encoder_dim", type=int, default=768, help="Dimension of the features from encoder")

    # --- 训练参数 ---
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_disentangler", type=float, default=1e-4)
    parser.add_argument("--lr_discriminator", type=float, default=2e-4)
    parser.add_argument("--grl_lambda", type=float, default=1.0, help="Weight for the Gradient Reverse Layer")
    parser.add_argument("--lambda_disc", type=float, default=0.5, help="Weight for the discrimination loss")
    parser.add_argument("--lambda_conf", type=float, default=0.5, help="Weight for the confusion loss")

    # ... (其他通用参数如device, num_workers, wandb设置等) ...

    args = parser.parse_args()
    train_stage2_disentangler(args)
