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

    # ... (wandb初始化) ...

    # --- 1. 加载并冻结第一阶段的自编码器 ---
    print("Loading and freezing the pretrained Stage 1 autoencoder...")

    # a. 初始化一个空的FomoJointAutoencoder结构
    config = ...
    fomo_encoder = MultiSpectralViT(...)
    autoencoder = FomoJointAutoencoder(fomo_encoder=fomo_encoder, ...)

    # b. 加载第一阶段训练好的权重
    autoencoder.load_state_dict(torch.load(args.stage1_ckpt_path, map_location='cpu'))

    # c. 冻结所有参数
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.to(device)
    print("Stage 1 autoencoder is loaded and frozen.")

    # --- 2. 初始化需要训练的新模型 ---
    print("Initializing Stage 2 models (Disentangler, Discriminator, Confuser)...")
    disentangler = FeatureDisentangler(feature_dim=args.encoder_dim).to(device)
    discriminator_related = ModalityDiscriminator(feature_dim=args.encoder_dim).to(device)
    confuser_unrelated = ModalityDiscriminator(feature_dim=args.encoder_dim).to(device)  # 结构相同

    # --- 3. 数据准备 (需要模态标签) ---
    print("Preparing datasets with modality labels...")
    # ... (与第一阶段类似, 但需要给Dataset传入modality_label) ...

    sar_dataset = FomoCompatibleDataset(..., modality_label=0)
    opt_dataset = FomoCompatibleDataset(..., modality_label=1)

    sar_loader = DataLoader(sar_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=custom_collate_fn)
    opt_loader = DataLoader(opt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=custom_collate_fn)

    # --- 4. 优化器和损失函数 ---
    optimizer_disentangler = torch.optim.AdamW(disentangler.parameters(), lr=args.lr_disentangler)
    optimizer_discriminator = torch.optim.AdamW(discriminator_related.parameters(), lr=args.lr_discriminator)
    optimizer_confuser = torch.optim.AdamW(confuser_unrelated.parameters(), lr=args.lr_discriminator)
    criterion_cls = nn.CrossEntropyLoss()

    print("Starting Stage 2: Feature Disentanglement Training...")
    for epoch in range(args.num_epochs):
        disentangler.train()
        discriminator_related.train()
        confuser_unrelated.train()
        # ... (数据迭代器和tqdm设置) ...
        for i in range(num_batches):
            # --- 获取SAR和OPT数据 ---
            sar_images, sar_keys = ...
            opt_images, opt_keys = ...

            # --- 提取冻结的混合特征 ---
            with torch.no_grad():
                _, mixed_tokens_sar = autoencoder((sar_images, sar_keys), modality='sar')
                _, mixed_tokens_opt = autoencoder((opt_images, opt_keys), modality='opt')

            mixed_tokens = torch.cat([mixed_tokens_sar, mixed_tokens_opt], dim=0)

            # --- 创建模态标签 ---
            b_sar, n_sar, d = mixed_tokens_sar.shape
            b_opt, n_opt, d = mixed_tokens_opt.shape
            labels_sar = torch.zeros(b_sar * n_sar, dtype=torch.long, device=device)
            labels_opt = torch.ones(b_opt * n_opt, dtype=torch.long, device=device)
            modality_labels_flat = torch.cat([labels_sar, labels_opt])

            # ============================================
            #  (1) 训练解耦器 (Disentangler)
            # ============================================
            optimizer_disentangler.zero_grad()

            unrelated_tokens, related_tokens = disentangler(mixed_tokens)

            # a. 判别损失 (希望related_tokens能被分开)
            pred_related = discriminator_related(related_tokens)
            loss_disc = criterion_cls(pred_related, modality_labels_flat)

            # b. 混淆损失 (希望unrelated_tokens不能被分开)
            reversed_unrelated = GradientReverseLayer.apply(unrelated_tokens, args.grl_lambda)
            pred_unrelated = confuser_unrelated(reversed_unrelated)
            loss_conf = criterion_cls(pred_unrelated, modality_labels_flat)

            total_loss_disentangler = args.lambda_disc * loss_disc + args.lambda_conf * loss_conf
            total_loss_disentangler.backward()
            optimizer_disentangler.step()

            # ============================================
            #  (2) 训练判别器 (Discriminator)
            # ============================================
            optimizer_discriminator.zero_grad()

            # # .detach()来避免梯度传回解耦器
            _, related_tokens_detached = disentangler(mixed_tokens.detach())
            pred_related_for_disc = discriminator_related(related_tokens_detached.detach())
            loss_disc_only = criterion_cls(pred_related_for_disc, modality_labels_flat)
            loss_disc_only.backward()
            optimizer_discriminator.step()

            # ============================================
            #  (3) 训练混淆器 (Confuser)
            # ============================================
            optimizer_confuser.zero_grad()

            unrelated_tokens_detached, _ = disentangler(mixed_tokens.detach())
            pred_unrelated_for_conf = confuser_unrelated(unrelated_tokens_detached.detach())
            loss_conf_only = criterion_cls(pred_unrelated_for_conf, modality_labels_flat)
            loss_conf_only.backward()
            optimizer_confuser.step()

        # ... (日志记录) ...

        print("Stage 2 feature disentanglement training complete!")
    # 保存解耦器模型
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
