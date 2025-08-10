import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import random
import argparse
from tqdm import tqdm
import shutil
import glob
import matplotlib.pyplot as plt

# ==============================================================================
# 导入训练时使用的模型类和函数
# ==============================================================================
from models.conditional_mdn import ConditionalMDN
from diffusers import DDPMScheduler


# UNetModel的定义 (这里假设它和你的训练脚本在同一文件或路径中)
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
        batch_size = x.shape[0]
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


# 导入训练脚本中的辅助函数
def set_determinism(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_psnr(img1, img2, data_range=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(img1, img2):
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


# 导入训练脚本中使用的 SAR2OptDataset 类
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
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
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
            sar_tensor = self.transform(sar_img)
            opt_tensor = self.transform(opt_img)
            return {"sar": sar_tensor, "opt": opt_tensor}
        except Exception as e:
            print(f"加载图像出错: {sar_path} 或 {opt_path}, 错误: {e}")
            return {"sar": torch.zeros(1, self.image_size, self.image_size),
                    "opt": torch.zeros(3, self.image_size, self.image_size)}


def get_train_val_dataset(data_dir, image_size=512):
    train_ds = SAR2OptDataset(root_dir=data_dir, image_size=image_size, is_train=True)
    val_ds = SAR2OptDataset(root_dir=data_dir, image_size=image_size, is_train=False)
    return train_ds, val_ds, val_ds


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_determinism(42)

data_dir = "/home/wushixuan/yujun/data/rsdiffusion/sar2opt"
logdir = f"./logs_sar2opt/diffusion_240_sar_to_opt"
model_save_path = os.path.join(logdir, "model")
image_size = 512
batch_size = 1
device = "cuda:0"


class TestTrainer:
    def __init__(self, model, c_mdn, sar_encoder, device, num_samples):
        self.device = device
        self.model = model.to(device).eval()
        self.c_mdn = c_mdn.to(device).eval()
        self.sar_encoder = sar_encoder.to(device).eval()
        self.num_samples = num_samples
        self.index = 0
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2"
        )

        # 确保所有模型参数都已冻结
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.c_mdn.parameters():
            param.requires_grad = False
        for param in self.sar_encoder.parameters():
            param.requires_grad = False

    def cal_metric(self, pred, gt):
        # 计算 PSNR/SSIM 前，将数据范围从 [0, 1] 转换为 [0, 255]
        pred_norm = pred.clamp(0, 1)
        gt_norm = gt.clamp(0, 1)

        pred_np = (pred_norm.cpu().numpy()[0] * 255).astype(np.uint8)
        gt_np = (gt_norm.cpu().numpy()[0] * 255).astype(np.uint8)

        psnr = compute_psnr(pred_np, gt_np, data_range=255.0)
        ssim = compute_ssim(pred_np, gt_np)

        mae = np.mean(np.abs(pred_np.astype(np.float64) - gt_np.astype(np.float64)))
        return psnr, ssim, mae

    def test(self, test_loader):
        psnr_list, ssim_list, mae_list = [], [], []

        test_dir = "./test_results_sar2opt/"
        shutil.rmtree(test_dir, ignore_errors=True)
        os.makedirs(test_dir, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                sar_images = batch["sar"].to(self.device)
                opt_images = batch["opt"].to(self.device)

                # --- 核心推理逻辑：与训练时保持一致 ---
                # 1. 编码 SAR 图像
                sar_feat_vector = self.sar_encoder(sar_images)

                # 2. 从 MDN 中采样特征
                mdn_feature = torch.randn(sar_feat_vector.shape[0], 1, self.c_mdn.feature_dim).to(self.device)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                          (sar_feat_vector.shape[0],), device=self.device).long()
                sar_feat_vector_3d = sar_feat_vector.unsqueeze(1)

                epsilon_pred = self.c_mdn(
                    noisy_style_tokens=mdn_feature,
                    time_steps=timesteps,
                    content_condition_tokens=sar_feat_vector_3d
                )
                sampled_rep = epsilon_pred.squeeze(1)

                # 3. UNet 前向传播
                pred = self.model(sar_images, sampled_rep)
                # --- 推理逻辑结束 ---

                # 计算指标
                psnr, ssim, mae = self.cal_metric(pred, opt_images)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                mae_list.append(mae)

                # 保存图像
                self.save_images(sar_images, opt_images, pred, test_dir)

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_mae = np.mean(mae_list)

        print("\n" + "=" * 50)
        print(f"测试汇总 | 平均PSNR: {avg_psnr:.4f}, 平均SSIM: {avg_ssim:.4f}, 平均MAE: {avg_mae:.4f}")
        print("=" * 50)

    def save_images(self, sar, gt, pred, save_dir):
        # 批量保存图像
        for i in range(sar.shape[0]):
            filename = f"{self.index:04d}"

            # 预测图像 (RGB)
            pred_tensor = pred[i].cpu().clamp(0, 1)
            pred_save = (pred_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_save).save(os.path.join(save_dir, f"{filename}_pred.png"))

            # 真实图像 (RGB)
            gt_tensor = gt[i].cpu().clamp(0, 1)
            gt_save = (gt_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(gt_save).save(os.path.join(save_dir, f"{filename}_gt.png"))

            # SAR输入图像 (Gray)
            sar_tensor = sar[i].cpu().clamp(0, 1)
            sar_save = (sar_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
            Image.fromarray(sar_save).save(os.path.join(save_dir, f"{filename}_sar.png"))

            self.index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Stage 4 Simplified UNet Model")
    parser.add_argument("--unet_ckpt_path", type=str, required=True, help="Path to the trained UNet checkpoint")
    parser.add_argument("--mdn_ckpt_path", type=str, default='./checkpoints_stage3_cmdn/c_mdn_best.pt',
                        help="Path to the trained MDN checkpoint")
    parser.add_argument("--data_dir", type=str, default='/home/wushixuan/yujun/data/rsdiffusion/sar2opt/',
                        help="Path to the dataset directory")
    parser.add_argument("--image_size", type=int, default=512, help="Image size for testing")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    args = parser.parse_args()

    # --- 设置参数 ---
    unet_ckpt_path = args.unet_ckpt_path
    mdn_ckpt_path = args.mdn_ckpt_path
    data_dir = args.data_dir
    image_size = args.image_size
    batch_size = args.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 模型配置 ---
    sar_channels = 1
    opt_channels = 3
    mdn_feature_dim = 768
    mdn_condition_dim = 768

    # --- 加载模型 ---
    print("加载模型...")

    # 1. 加载UNet模型
    unet = UNetModel(
        image_size=image_size,
        in_channels=sar_channels,
        model_channels=256,
        out_channels=opt_channels,
        condition_channels=mdn_feature_dim
    )
    unet.load_state_dict(torch.load(unet_ckpt_path, map_location='cpu'))

    # 2. 加载MDN模型
    c_mdn = ConditionalMDN(
        feature_dim=mdn_feature_dim, condition_dim=mdn_condition_dim
    )
    c_mdn.load_state_dict(torch.load(mdn_ckpt_path, map_location='cpu'))

    # 3. 初始化SAR编码器
    sar_encoder = nn.Sequential(
        nn.Conv2d(1, 64, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(256, 512, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(512, mdn_condition_dim, 4, 2, 1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )

    # --- 创建训练器并运行测试 ---
    trainer = TestTrainer(
        model=unet,
        c_mdn=c_mdn,
        sar_encoder=sar_encoder,
        device=device,
        num_samples=1  # 测试时通常只需要一次采样
    )

    print("\n开始测试SAR2Opt模型...")
    test_ds = SAR2OptDataset(root_dir=data_dir, image_size=image_size, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"测试集样本数: {len(test_ds)}")

    trainer.test(test_loader)