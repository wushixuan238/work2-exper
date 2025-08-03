# train_stage3_conditional_mdn.py

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 导入所有必要的模块 ---
from models.fomo_joint_autoencoder import FomoJointAutoencoder
from FoMo.model_zoo.multimodal_mae import MultiSpectralViT
from models.disentangler import FeatureDisentangler
from models.conditional_mdn import ConditionalMDN, get_timestep_embedding  # 导入新模型
from dataset.fomo_dataset import FomoCompatibleDataset


# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler # 可以使用diffusers库来管理扩散过程

def train_stage3_conditional_mdn(args):
    device = torch.device(args.device)

    # --- 1. 加载并冻结第一、二阶段的模型 ---
    print("Loading and freezing pretrained models from Stage 1 & 2...")
    # a. 加载自编码器 (Stage 1)
    # ... (与第二阶段加载逻辑相同) ...
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

    # b. 加载解耦器 (Stage 2)
    disentangler = FeatureDisentangler(...)
    disentangler.load_state_dict(torch.load(args.stage2_ckpt_path))
    disentangler.eval().to(device)
    print("All pretrained models are loaded and frozen.")

    # --- 2. 初始化需要训练的新模型：C-MDN ---
    print("Initializing Stage 3 model (Conditional MDN)...")
    c_mdn = ConditionalMDN(
        feature_dim=args.encoder_dim,
        condition_dim=args.encoder_dim,  # 内容和风格特征维度相同
        # ... 其他超参数 ...
    ).to(device)

    # --- 3. 数据准备 (只需要光学数据) ---
    print("Preparing Optical dataset for training...")
    transform_sar = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(num_output_channels=3),  # !! 将灰度图转换为3通道 !!
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

    # --- 4. 设置扩散过程 ---
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # --- 5. 优化器和损失函数 ---
    optimizer = torch.optim.AdamW(c_mdn.parameters(), lr=args.learning_rate)
    criterion_mse = nn.MSELoss()

    print("Starting Stage 3: Conditional MDN Training...")
    for epoch in range(args.num_epochs):
        c_mdn.train()

        for images, keys in tqdm(dataloader, ...):
            images = images.to(device)

        # --- a. 提取冻结的特征 ---
        with torch.no_grad():
            mixed_tokens_opt = autoencoder.encoder((images, keys), pool=False)
            unrelated_opt, related_opt_0 = disentangler(mixed_tokens_opt)

        # --- b. 准备扩散模型输入(严格遵循 Epsilon-Prediction) ---

        # 1. 采样噪声 epsilon (ε)
        # epsilon 的形状与干净的风格特征完全相同
        epsilon = torch.randn_like(related_opt_0)
        bsz = related_opt_0.shape[0]

        # 2. 采样时间步
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

        # 3. 对干净的风格特征加噪，得到 xₜ
        # noise_scheduler.add_noise 会自动处理alpha, beta等复杂的系数
        noisy_related_opt_t = noise_scheduler.add_noise(related_opt_0, epsilon, timesteps)

        # --- c. 训练C-MDN ---
        optimizer.zero_grad()

        # 1. 模型预测 epsilon_pred
        # 模型的输入是带噪声的数据 xₜ，时间步 t，以及条件 c
        epsilon_pred = c_mdn(
            noisy_style_tokens=noisy_related_opt_t,
            time_steps=timesteps,  # 直接传入整数时间步，模型内部处理嵌入
            content_condition_tokens=unrelated_opt
        )

        # 2. 计算损失 (预测的epsilon vs 真实的epsilon)
        # 这就是标准的epsilon-prediction loss
        loss = criterion_mse(epsilon_pred, epsilon)

        loss.backward()
        optimizer.step()

        # ============================================
        #            验证循环 (Validation Loop)
        # ============================================
        c_mdn.eval()
        val_loss_epoch = 0

        with torch.no_grad():  # 验证时不需要计算梯度
            for images, keys in tqdm(val_loader, desc=f"Stage 3 - Validation Epoch {epoch + 1}"):
        # ... (验证逻辑与训练逻辑几乎一样，但不进行优化) ...

        # a. 提取特征
        # with torch.no_grad():
        #     ...
        #     unrelated_opt, related_opt_0 = ...

        # b. 加噪 (使用相同的逻辑)
        # epsilon = torch.randn_like(related_opt_0)
        # timesteps = torch.randint(...)
        # noisy_related_opt_t = noise_scheduler.add_noise(...)

        # c. 预测epsilon
        # epsilon_pred = c_mdn(...)

        # d. 计算损失
        # loss = criterion_mse(epsilon_pred, epsilon)
        # val_loss_epoch += loss.item()

        avg_val_loss = val_loss_epoch / len(val_loader)

        print(f"Epoch {epoch + 1}/{args.num_epochs} | "
              f"Avg Train Loss: {avg_train_loss:.4f} | "
              f"Avg Val Loss: {avg_val_loss:.4f}")

        # ... (wandb 日志记录 avg_train_loss 和 avg_val_loss) ...

        # ============================================
        #       模型保存逻辑 (核心修改)
        # ============================================

        # 1. 保存当前epoch的模型 (可选，用于断点续训)
        # current_ckpt_path = os.path.join(args.output_dir, "c_mdn_latest.pt")
        # torch.save(c_mdn.state_dict(), current_ckpt_path)

        # 2. 检查并保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(args.output_dir, "c_mdn_best.pt")
            torch.save(c_mdn.state_dict(), best_ckpt_path)
            print(f"🎉 New best model saved to: {best_ckpt_path} with Val Loss: {best_val_loss:.4f}")

    print("Stage 3 Conditional MDN training complete!")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")

    # ... (日志记录) ...

    print("Stage 3 Conditional MDN training complete!")
    # 保存模型
    output_path = os.path.join(args.output_dir, "c_mdn_stage3.pt")
    torch.save(c_mdn.state_dict(), output_path)
    print(f"Stage 3 C-MDN model saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 3: Train a Conditional Modality Distribution Network")

    # --- 路径参数 ---
    parser.add_argument("--stage1_ckpt_path", type=str, required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--stage2_ckpt_path", type=str, required=True, help="Path to Stage 2 checkpoint")
    # ... 其他路径和配置参数 ...

    # --- 模型参数 ---
    # ... (需要encoder_dim等与前两阶段匹配的参数) ...

    # --- 训练参数 ---
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # ... 其他通用参数 ...

    args = parser.parse_args()
    train_stage3_conditional_mdn(args)
