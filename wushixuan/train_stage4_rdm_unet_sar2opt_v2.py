import numpy as np
import torch
import torch.nn as nn
import random
import os
import sys
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import time
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import argparse

# ==============================================================================
# 导入所需的模型类
# ==============================================================================
from models.conditional_mdn import ConditionalMDN
from diffusers import DDPMScheduler

# 新增：导入 FoMo 模型的类
from FoMo.model_zoo.multimodal_mae import MultiSpectralViT
from wushixuan.train_stage1_sar2opt import FomoJointAutoencoder


# --- UNetModel 和其他辅助函数 ---
def set_determinism(seed=None):
    """设置随机种子以保证结果可复现"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_new_model_and_delete_last(model, save_path, delete_symbol=None):
    """保存新模型并删除旧模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")
    if delete_symbol is not None:
        dir_path = os.path.dirname(save_path)
        for file in os.listdir(dir_path):
            if delete_symbol in file and file != os.path.basename(save_path):
                os.remove(os.path.join(dir_path, file))
                print(f"已删除旧模型: {file}")


def compute_psnr(img1, img2, data_range=1.0):
    """计算峰值信噪比 (PSNR)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(img1, img2):
    """计算结构相似性 (SSIM)"""
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.mean((img1 - mu1) ** 2)
    sigma2_sq = np.mean((img2 - mu2) ** 2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


class UNetModel(nn.Module):
    """
    一个简化的UNet模型，输入是SAR图像，条件是MDN采样的特征向量。
    """

    def __init__(self, image_size, in_channels, model_channels, out_channels, condition_channels):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels

        self.conv_in = nn.Conv2d(self.in_channels, model_channels, kernel_size=3, padding=1)
        self.condition_proj = nn.Linear(self.condition_channels, model_channels)

        self.down1 = nn.Sequential(
            nn.Conv2d(model_channels, model_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(model_channels * 2, model_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(model_channels * 4, model_channels * 4, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(model_channels * 4, model_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(model_channels * 2, model_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv_out = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, condition=None):
        h = self.conv_in(x)

        if condition is not None:
            cond_proj = self.condition_proj(condition)
            h = h + cond_proj.unsqueeze(-1).unsqueeze(-1)

        h = self.down1(h)
        h = self.down2(h)
        h = self.middle(h)
        h = self.up1(h)
        h = self.up2(h)

        output = self.conv_out(h)
        return torch.sigmoid(output)


# ---
# BaseTrainer 类 (已修改)
# ---

class BaseTrainer:
    def __init__(self, model, max_epochs, learning_rate, weight_decay, log_dir, save_path,
                 device, batch_size, num_samples, c_mdn, sar_encoder, **kwargs):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion_mse = nn.MSELoss().to(device)
        self.max_epochs = max_epochs
        self.log_dir = log_dir
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.val_every = 1
        self.best_score = 0.0
        self.global_step = 0
        self.current_epoch = 0
        self.log_data = {}

        # 新增：使用传入的 FoMo 编码器作为 sar_encoder
        print("Using FoMo encoder as sar_encoder...")
        self.sar_encoder = sar_encoder.to(device).eval()
        for param in self.sar_encoder.parameters():
            param.requires_grad = False

        # 加载并冻结 c_mdn 模型
        print("Initializing with pretrained c_mdn model...")
        self.c_mdn = c_mdn.to(device).eval()
        for param in self.c_mdn.parameters():
            param.requires_grad = False

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2"
        )
        self.mdn_feature_dim = self.c_mdn.feature_dim
        self.mdn_condition_dim = self.c_mdn.condition_dim

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        print(f"模型已移动到设备: {device}")

    def log(self, key, value, step=None):
        if step is None:
            step = self.global_step
        if key not in self.log_data:
            self.log_data[key] = []
        self.log_data[key].append((step, value))

    def train(self, train_dataset, val_dataset=None):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

        print(f"开始训练: {self.max_epochs} 轮, 每轮 {len(train_loader)} 批次")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self._train_epoch(train_loader)

            if val_dataset is not None and (epoch + 1) % self.val_every == 0:
                self._validate(val_loader)

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs}")

        for batch_idx, batch in enumerate(pbar):
            loss = self.training_step(batch)
            total_loss += loss.item()
            self.log(f"train_loss_step", loss.item())
            pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})
            self.global_step += 1

        print(f"Epoch {self.current_epoch + 1} 平均损失: {total_loss / len(train_loader):.4f}")

    def training_step(self, batch):
        self.optimizer.zero_grad()
        sar_images = batch["sar"].to(self.device)
        opt_images = batch["opt"].to(self.device)

        with torch.no_grad():
            sar_keys_hardcoded = [0]
            sar_tokens = self.sar_encoder((sar_images, sar_keys_hardcoded), pool=False)

            # 修正：FoMo 编码器是为多模态训练的，其输出包含了所有模态的 tokens。
            # 我们只需要 SAR 模态的 token，FoMo 的输出格式可能是 (tokens, masks)。
            # 你的 FoMo-ViT 的 configs 设为 37 模态，但你只传入一个模态，这会出错。
            # 解决办法：在初始化 FoMo 模型时，只配置 SAR 和 OPT 模态。

            # 修正：对 tokens 进行平均池化以得到一个 2D 特征向量
            sar_feat_vector = sar_tokens.mean(dim=1)  # [B, D]

            # 修正：为 c_mdn 准备 3D 形状的条件向量
            sar_feat_vector_3d = sar_feat_vector.unsqueeze(1)

            # 从MDN中采样特征
            mdn_feature = torch.randn(sar_feat_vector.shape[0], 1, self.mdn_feature_dim).to(self.device)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (sar_feat_vector.shape[0],),
                                      device=self.device).long()

            epsilon_pred = self.c_mdn(
                noisy_style_tokens=mdn_feature,
                time_steps=timesteps,
                content_condition_tokens=sar_feat_vector_3d
            )

            # 将 c_mdn 的输出从 [B, 1, D] 展平为 [B, D]
            sampled_rep = epsilon_pred.squeeze(1)

        pred = self.model(sar_images, sampled_rep)

        loss = self.criterion_mse(pred, opt_images)
        loss.backward()
        self.optimizer.step()

        self.log("loss", loss.item(), step=self.global_step)

        return loss

    def _validate(self, val_loader):
        self.model.eval()
        val_outputs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                sar_images = batch["sar"].to(self.device)
                opt_images = batch["opt"].to(self.device)

                psnr_vals = []
                ssim_vals = []

                for _ in range(self.num_samples):
                    sar_keys_hardcoded = [0]
                    sar_tokens = self.sar_encoder((sar_images, sar_keys_hardcoded), pool=False)
                    sar_feat_vector = sar_tokens.mean(dim=1)
                    sar_feat_vector_3d = sar_feat_vector.unsqueeze(1)

                    mdn_feature = torch.randn(sar_feat_vector.shape[0], 1, self.mdn_feature_dim).to(self.device)
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                              (sar_feat_vector.shape[0],), device=self.device).long()

                    epsilon_pred = self.c_mdn(
                        noisy_style_tokens=mdn_feature,
                        time_steps=timesteps,
                        content_condition_tokens=sar_feat_vector_3d
                    )
                    sampled_rep = epsilon_pred.squeeze(1)

                    pred = self.model(sar_images, sampled_rep)

                    pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    opt_np = opt_images.detach().cpu().numpy().transpose(0, 2, 3, 1)

                    psnr_vals.append(compute_psnr(pred_np[0], opt_np[0]))
                    ssim_vals.append(compute_ssim(pred_np[0], opt_np[0]))

                val_outputs.append({
                    "psnr": np.mean(psnr_vals),
                    "ssim": np.mean(ssim_vals)
                })
        self.model.train()
        self.validation_end(val_outputs)

    def validation_end(self, val_outputs):
        avg_psnr = np.mean([x["psnr"] for x in val_outputs])
        avg_ssim = np.mean([x["ssim"] for x in val_outputs])

        print(f"验证结果 - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

        self.log("psnr", avg_psnr, step=self.current_epoch)
        self.log("ssim", avg_ssim, step=self.current_epoch)

        if avg_psnr > self.best_score:
            self.best_score = avg_psnr
            print(f"发现更好的模型! 保存检查点... PSNR: {avg_psnr:.2f}")
            save_new_model_and_delete_last(self.model, os.path.join(self.save_path, f"best_model_{avg_psnr:.4f}.pt"),
                                           delete_symbol="best_model")


# --- SAR2OptDataset 类 (修正了 Normalize) ---
class SAR2OptDataset(Dataset):
    def __init__(self, root_dir, image_size=512, is_train=True):
        if is_train:
            self.sar_dir = os.path.join(root_dir, "trainA")
            self.opt_dir = os.path.join(root_dir, "trainB")
        else:
            self.sar_dir = os.path.join(root_dir, "testA")
            self.opt_dir = os.path.join(root_dir, "testB")

        self.sar_files = sorted(os.listdir(self.sar_dir))
        self.opt_files = sorted(os.listdir(self.opt_dir))
        assert len(self.sar_files) == len(self.opt_files), "数据集数量不匹配"
        self.image_size = image_size

        # 修正：为 SAR 和 OPT 分别定义不同的 transform
        self.transform_sar = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 针对单通道灰度图像的归一化
        ])

        self.transform_opt = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 针对三通道 RGB 图像的归一化
        ])

        print(f"已加载{len(self.sar_files)}对图像数据")

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar_filename = self.sar_files[idx]
        opt_filename = self.opt_files[idx]
        sar_path = os.path.join(self.sar_dir, sar_filename)
        opt_path = os.path.join(self.opt_dir, opt_filename)
        try:
            sar_img = Image.open(sar_path).convert('L')
            opt_img = Image.open(opt_path).convert('RGB')

            sar_tensor = self.transform_sar(sar_img)
            opt_tensor = self.transform_opt(opt_img)

            return {"sar": sar_tensor, "opt": opt_tensor}
        except Exception as e:
            print(f"加载图像出错: {sar_path} 或 {opt_path}, 错误: {e}")
            return {"sar": torch.zeros(1, self.image_size, self.image_size),
                    "opt": torch.zeros(3, self.image_size, self.image_size)}


def get_train_val_dataset(data_dir, image_size=512):
    train_ds = SAR2OptDataset(root_dir=data_dir, image_size=image_size, is_train=True)
    val_ds = SAR2OptDataset(root_dir=data_dir, image_size=image_size, is_train=False)
    return train_ds, val_ds, val_ds


# ---
# Main Function (已修改)
# ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Simplified UNet Training with RDM-MDN Conditions")
    parser.add_argument("--mdn_ckpt_path", type=str, default='./checkpoints_stage3_cmdn/c_mdn_best.pt',
                        help="Path to the trained MDN checkpoint")
    parser.add_argument("--fomo_ckpt_path", type=str,
                        default='/home/wushixuan/桌面/07/work2-exper/wushixuan/checkpoints_stage1_optical/fomo_joint_autoencoder_stage1.pt',
                        help="Path to the trained FoMo autoencoder checkpoint")
    parser.add_argument("--data_dir", type=str, default='/home/wushixuan/yujun/data/rsdiffusion/sar2opt/',
                        help="Path to the dataset directory")
    parser.add_argument("--image_size", type=int, default=512, help="Image size for training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for FoMo encoder")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    set_determinism(seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("加载预训练模型...")
    mdn_ckpt_path = args.mdn_ckpt_path
    fomo_ckpt_path = args.fomo_ckpt_path
    data_dir = args.data_dir
    image_size = args.image_size
    patch_size = args.patch_size

    # 定义模型参数
    sar_channels = 1
    opt_channels = 3
    mdn_feature_dim = 768
    mdn_condition_dim = 768

    # 加载 Stage 3 MDN 模型
    c_mdn = ConditionalMDN(feature_dim=mdn_feature_dim, condition_dim=mdn_condition_dim)
    c_mdn.load_state_dict(torch.load(mdn_ckpt_path, map_location='cpu'))
    print("c_mdn模型加载成功。")

    # 新增：加载 FoMo 编码器作为 sar_encoder
    print("加载 FoMo 编码器...")
    # 修正：FoMo 编码器的配置需要与训练时使用的通道数一致
    fomo_init_configs = {
        'single_embedding_layer': True,
        'modality_channels': list(range(37))
    }

    fomo_encoder = MultiSpectralViT(
        image_size=image_size, patch_size=patch_size, channels=1, num_classes=1000,
        dim=mdn_condition_dim, depth=12, heads=12, mlp_dim=2048,
        configs=fomo_init_configs
    )
    autoencoder = FomoJointAutoencoder(
        fomo_encoder=fomo_encoder, decoder_dim=512, decoder_depth=8,
        sar_channels=sar_channels, opt_channels=opt_channels,
        image_size=image_size, patch_size=patch_size
    )

    # 修正：加载权重时，如果存在维度不匹配，跳过这些参数
    try:
        autoencoder.load_state_dict(torch.load(fomo_ckpt_path, map_location='cpu'), strict=False)
        print("FoMo 编码器加载成功。")
    except RuntimeError as e:
        print(f"加载 FoMo 模型权重时出错: {e}")
        # 如果这里仍然失败，那说明 FoMo 的架构配置与检查点严重不符
        sys.exit(1)

    sar_encoder = autoencoder.encoder

    # 创建UNet模型
    unet = UNetModel(
        image_size=image_size,
        in_channels=sar_channels,
        model_channels=256,
        out_channels=opt_channels,
        condition_channels=mdn_feature_dim
    )
    unet = unet.to(device)
    print("UNet模型已创建并移动到设备。")

    # 创建训练器，并将 c_mdn 模型作为参数传入
    trainer = BaseTrainer(
        model=unet,
        max_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=1e-5,
        log_dir="./logs_sar2opt/rdm_unet_fomo/log",
        save_path="./logs_sar2opt/rdm_unet_fomo/model",
        device=device,
        batch_size=args.batch_size,
        num_samples=1,
        c_mdn=c_mdn,
        sar_encoder=sar_encoder,
    )

    # 获取训练集和验证集
    train_ds, val_ds, _ = get_train_val_dataset(data_dir=data_dir, image_size=image_size)

    # 开始训练
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)