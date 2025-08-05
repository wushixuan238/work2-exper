import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from diffusers import DDPMScheduler
from tqdm import tqdm
import wandb

from FoMo.model_zoo.multimodal_mae import MultiSpectralViT
from models.conditional_mdn import ConditionalMDN
from models.disentangler import FeatureDisentangler
from wushixuan.train_stage1_sar2opt import FomoJointAutoencoder
from yujun.dataset.fomo_dataset import FomoCompatibleDataset


def train_stage3_conditional_mdn(args):
    device = torch.device(args.device)

    opt_keys_hardcoded = [0, 1, 2]  # 假设模型可以处理3个通道
    sar_keys_hardcoded = [0]  # 假设sar通道是1

    # 1. 初始化wandb
    if args.use_wandb:
        wandb.init(project=args.project_name, name="stage3_conditional_mdn", config=args)

    # 2. 加载配置文件，为模型和数据加载做准备
    print(f"Loading config from {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # 3. 加载并冻结第一、二阶段的模型
    print("Loading and freezing pretrained models from Stage 1 & 2...")

    # a. 根据配置文件参数构建 Stage 1 模型
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

    # b. 加载 Stage 1 训练好的权重并冻结
    print(f"Loading Stage 1 checkpoint from: {args.stage1_ckpt_path}")
    autoencoder.load_state_dict(torch.load(args.stage1_ckpt_path, map_location='cpu'))
    autoencoder.eval().to(device)
    for param in autoencoder.parameters():
        param.requires_grad = False

    # c. 加载 Stage 2 解耦器并冻结
    disentangler = FeatureDisentangler(feature_dim=args.encoder_dim)
    print(f"Loading Stage 2 checkpoint from: {args.stage2_ckpt_path}")
    disentangler.load_state_dict(torch.load(args.stage2_ckpt_path, map_location='cpu'))
    disentangler.eval().to(device)
    for param in disentangler.parameters():
        param.requires_grad = False

    print("All pretrained models are loaded and frozen.")

    # 4. 初始化需要训练的新模型：C-MDN
    print("Initializing Stage 3 model (Conditional MDN)...")
    c_mdn = ConditionalMDN(
        feature_dim=args.encoder_dim,
        condition_dim=args.encoder_dim,
    ).to(device)

    # 5. 数据准备
    print("Preparing dataset for training...")
    # 这里只使用光学数据集来训练C-MDN，因为它是在OPT特征空间进行学习
    transform_opt = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    opt_map = config['dataset_modality_index']['my_optical_train_dataset']
    sorted_opt_map = sorted(opt_map.items(), key=lambda item: item[1])
    opt_band_names = [item[0] for item in sorted_opt_map]
    band_name_to_key = {name: int(key) for key, name in config['modality_channels'].items()}
    # opt_keys = [band_name_to_key[name] for name in opt_band_names]
    # 我们不使用这个，我们使用硬编码的

    # 你的模型架构是基于 pretraining 的配置来的
    # 既然报错 `size 2`，这意味着 pretraining 时的 `num_spectral` 是 2
    # 所以我们必须把keys映射到 [0, 1]
    opt_keys_for_model = [0, 1]
    sar_keys_for_model = [0]

    opt_dataset = FomoCompatibleDataset(
        data_dir=args.opt_data_dir, dataset_name='my_optical_train_dataset',
        config=config, transform=transform_opt, in_chans=args.opt_channels, modality_label=1
    )

    dataloader = DataLoader(opt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # 这里我们简化，不创建单独的验证集，但在实际应用中应该创建
    # 假设 val_dataset 也是一个 DataLoader
    val_dataset = FomoCompatibleDataset(
        data_dir='/home/wushixuan/yujun/data/rsdiffusion/sar2opt/testA', dataset_name='my_optical_train_dataset',
        config=config, transform=transform_opt, in_chans=args.opt_channels, modality_label=1
    )  # 简化处理
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 6. 设置扩散过程
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2"
    )

    # 7. 优化器和损失函数
    optimizer = torch.optim.AdamW(c_mdn.parameters(), lr=args.learning_rate)
    criterion_mse = nn.MSELoss()

    best_val_loss = float('inf')

    # 8. 训练和验证循环
    print("Starting Stage 3: Conditional MDN Training...")
    for epoch in range(args.num_epochs):
        c_mdn.train()
        avg_train_loss = 0

        # 训练循环
        with tqdm(dataloader, desc=f"Stage 3 - Epoch {epoch + 1}/{args.num_epochs}") as pbar:
            for images, keys in pbar:
                images = images.to(device)
                hardcoded_keys = [0, 1,2]

                # a. 提取冻结的特征
                with torch.no_grad():
                    mixed_tokens_opt = autoencoder.encoder((images, hardcoded_keys), pool=False)
                    # disnetangler 期望一个包含所有token的批次
                    unrelated_opt, related_opt_0 = disentangler(mixed_tokens_opt)

                # b. 准备扩散模型输入
                epsilon = torch.randn_like(related_opt_0)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (related_opt_0.shape[0],),
                                          device=device).long()
                noisy_related_opt_t = noise_scheduler.add_noise(related_opt_0, epsilon, timesteps)

                # c. 训练C-MDN
                optimizer.zero_grad()
                epsilon_pred = c_mdn(
                    noisy_style_tokens=noisy_related_opt_t,
                    time_steps=timesteps,
                    content_condition_tokens=unrelated_opt
                )
                loss = criterion_mse(epsilon_pred, epsilon)
                loss.backward()
                optimizer.step()

                avg_train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        avg_train_loss /= len(dataloader)

        # 验证循环
        c_mdn.eval()
        avg_val_loss = 0
        with torch.no_grad():
            for images, keys in tqdm(val_loader, desc=f"Stage 3 - Validation Epoch {epoch + 1}"):
                images = images.to(device)
                hardcoded_keys = [0, 1, 2]
                # print(f"DEBUG: Current batch images shape: {images.shape}")
                # print(f"DEBUG: Current batch keys: {keys}")
                #
                # # 在这里检查 keys 列表的长度
                # if len(keys[0]) != images.shape[1]:
                #     print(f"警告：Keys列表长度 {len(keys[0])} 与图像通道数 {images.shape[1]} 不匹配！")
                #     # 可以在这里跳过本次循环，或者采取其他措施
                #     continue

                # a. 提取特征
                mixed_tokens_opt = autoencoder.encoder((images, hardcoded_keys), pool=False)
                unrelated_opt, related_opt_0 = disentangler(mixed_tokens_opt)

                # b. 加噪
                epsilon = torch.randn_like(related_opt_0)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (related_opt_0.shape[0],),
                                          device=device).long()
                noisy_related_opt_t = noise_scheduler.add_noise(related_opt_0, epsilon, timesteps)

                # c. 预测 epsilon
                epsilon_pred = c_mdn(
                    noisy_style_tokens=noisy_related_opt_t,
                    time_steps=timesteps,
                    content_condition_tokens=unrelated_opt
                )

                loss = criterion_mse(epsilon_pred, epsilon)
                avg_val_loss += loss.item()

        avg_val_loss /= len(val_loader)

        # 9. 打印和保存
        print(f"Epoch {epoch + 1}/{args.num_epochs} | "
              f"Avg Train Loss: {avg_train_loss:.4f} | "
              f"Avg Val Loss: {avg_val_loss:.4f}")

        if args.use_wandb:
            wandb.log({"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss, "epoch": epoch})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(args.output_dir, "c_mdn_best.pt")
            torch.save(c_mdn.state_dict(), best_ckpt_path)
            print(f"🎉 New best model saved to: {best_ckpt_path} with Val Loss: {best_val_loss:.4f}")

    print("Stage 3 Conditional MDN training complete!")
    output_path = os.path.join(args.output_dir, "c_mdn_stage3.pt")
    torch.save(c_mdn.state_dict(), output_path)
    print(f"Final Stage 3 C-MDN model saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 3: Train a Conditional Modality Distribution Network")

    # --- 路径参数 ---
    parser.add_argument("--stage1_ckpt_path", type=str,
                        default='/home/wushixuan/桌面/07/work2-exper/wushixuan/checkpoints_stage1_optical/fomo_joint_autoencoder_stage1.pt',
                        help="Path to the trained Stage 1 autoencoder checkpoint")
    parser.add_argument("--stage2_ckpt_path", type=str,
                        default='/home/wushixuan/桌面/07/work2-exper/wushixuan/checkpoints_stage2_disentangler/disentangler_stage2.pt',
                        help="Path to Stage 2 checkpoint")
    parser.add_argument("--opt_data_dir", type=str, default='/home/wushixuan/yujun/data/rsdiffusion/sar2opt/trainA',
                        help="Path to Optical data directory")
    parser.add_argument("--sar_data_dir", type=str, default='/home/wushixuan/yujun/data/rsdiffusion/sar2opt/trainB',
                        help="Path to SAR data directory")
    parser.add_argument("--config_path", type=str,
                        default='/home/wushixuan/桌面/07/work2-exper/FoMo/configs/datasets/fomo_pretraining_datasets.json',
                        help="Path to FoMo-Net config file")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage3_cmdn")

    # --- 模型参数 ---
    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--encoder_heads", type=int, default=12)
    parser.add_argument("--encoder_mlp_dim", type=int, default=2048)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_depth", type=int, default=8)
    parser.add_argument("--decoder_heads", type=int, default=16)

    # --- 数据参数 ---
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--sar_channels", type=int, default=1)
    parser.add_argument("--opt_channels", type=int, default=3)

    # --- 训练参数 ---
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_wandb", action='store_true', help="Use wandb for logging")
    parser.add_argument("--project_name", type=str, default="SAR-to-Optical-Translation")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_stage3_conditional_mdn(args)
