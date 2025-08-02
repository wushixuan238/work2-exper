import argparse
import os
import random
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Function

import wandb
from yujun.dataset.SAR2OptDataset import SAR2OptDataset, pair_Dataset
from guided_diffusion.unet_raw import UNetModel
from utils.patch_masking import mask_func


# 梯度反转层定义
class GradientReverseLayer(torch.autograd.Function):
    """梯度反转层，用于对抗训练"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReverseModule(nn.Module):
    """包装梯度反转层为模块"""

    def __init__(self, alpha=1.0):
        super(GradientReverseModule, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReverseLayer.apply(x, self.alpha)


def set_seed(seed):
    """设置随机种子，确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 添加梯度反转层
class ReverseGradFunction(Function):
    @staticmethod
    def forward(ctx, data, alpha=1.0):
        ctx.alpha = alpha
        return data

    @staticmethod
    def backward(ctx, grad_outputs):
        grad = None
        if ctx.needs_input_grad[0]:
            grad = -ctx.alpha * grad_outputs
        return grad, None


class ReverseGrad(nn.Module):
    """
    梯度反转层 (GRL): Gradient Reversed Layer
    用于对抗训练，反转梯度方向
    """

    def __init__(self):
        super(ReverseGrad, self).__init__()

    def forward(self, x, alpha=1.0):
        return ReverseGradFunction.apply(x, alpha)


class Discrimination(nn.Module):
    """
    模态判别器网络结构
    用于判断特征来自哪个模态(SAR或OPT)
    """

    def __init__(self, dim=2048):
        super(Discrimination, self).__init__()
        self.fc1 = nn.Linear(dim, 192)
        self.bn1 = nn.BatchNorm1d(192)
        self.fc2 = nn.Linear(192, 192)
        self.bn2 = nn.BatchNorm1d(192)
        self.fc3 = nn.Linear(192, 2)  # 二分类：SAR或OPT

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), training=self.training)
        x = self.fc3(x)
        return x


# 修改UNetModel以支持特征解耦
class IDKLUNetModel(nn.Module):
    """
    扩展UNetModel，支持模态无关和模态相关特征解耦
    """

    def __init__(self, base_model, h_dim=2048, h_channels=192, alpha=0.1):
        """
        初始化IDKL-UNet模型
        :param base_model: 基础UNetModel
        :param h_dim: 特征维度
        :param h_channels: 特征通道数
        :param alpha: 梯度反转层系数
        """
        super(IDKLUNetModel, self).__init__()
        self.base_model = base_model
        self.h_dim = h_dim
        self.h_channels = h_channels
        self.alpha = alpha

        # 将基础模型的输入/输出通道数保存为属性
        self.in_channels = base_model.in_channels
        self.out_channels = base_model.out_channels

        # 记录编码器输出通道数
        self.h_channels = h_channels
        self.h_dim = h_dim

        # 特征提取卷积层
        self.unrelated_branch = nn.Conv2d(h_channels, h_channels, kernel_size=3, padding=1)
        self.related_branch = nn.Conv2d(h_channels, h_channels, kernel_size=3, padding=1)

        # 特征投影层
        self.unrelated_proj = nn.Linear(h_channels * 8 * 8, h_dim)
        self.related_proj = nn.Linear(h_channels * 8 * 8, h_dim)

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        # 梯度反转层
        self.rev_layer = GradientReverseModule(alpha=alpha)

        # 预定义解码器所需层
        # 将融合特征调整为编码器输出通道数的适配层
        # 使用h_dim*2作为输入维度，因为我们连接了两个h_dim大小的特征
        self.feat_adapter = nn.Linear(h_dim * 2, h_channels)

        # 上采样层 - 将特征图从32x32恢复到256x256
        self.upsampler1 = nn.ConvTranspose2d(h_channels, h_channels // 2, kernel_size=4, stride=2, padding=1)
        self.upsampler2 = nn.ConvTranspose2d(h_channels // 2, h_channels // 4, kernel_size=4, stride=2, padding=1)
        self.upsampler3 = nn.ConvTranspose2d(h_channels // 4, h_channels // 8, kernel_size=4, stride=2, padding=1)

        # 简化版解码器 - 从上采样后的特征图到输出通道数
        self.simple_decoder = nn.Conv2d(h_channels // 8, base_model.out_channels, kernel_size=3, padding=1)

        # 初始化梯度反转层
        # 使用前面定义的GradientReverseModule
        self.grl = GradientReverseModule(alpha=alpha)

    def encode(self, x):
        """
        编码阶段：提取模态无关和模态相关特征
        """
        # 使用基础模型的编码器部分提取深层特征
        # print(f"[encode] 输入形状: {x.shape}")
        h = x
        for module in self.base_model.input_blocks:
            h = module(h)
        h = self.base_model.middle_block(h)
        # print(f"[encode] 编码器输出形状: {h.shape}")

        # 提取模态无关特征
        unrelated_feat = self.unrelated_branch(h)
        # print(f"[encode] 模态无关分支输出: {unrelated_feat.shape}")
        unrelated_feat = F.adaptive_avg_pool2d(unrelated_feat, (1, 1))
        unrelated_feat = unrelated_feat.view(unrelated_feat.size(0), -1)  # 展平
        # print(f"[encode] 展平后模态无关特征: {unrelated_feat.shape}")

        # 提取模态相关特征
        related_feat = self.related_branch(h)
        # print(f"[encode] 模态相关分支输出: {related_feat.shape}")
        related_feat = F.adaptive_avg_pool2d(related_feat, (1, 1))
        related_feat = related_feat.view(related_feat.size(0), -1)  # 展平
        # print(f"[encode] 展平后模态相关特征: {related_feat.shape}")

        return unrelated_feat, related_feat, h

    def decode(self, y0_unrelated, y0_related, h_deep):
        """
        解码阶段：将解耦特征重新组合并解码为图像
        """
        # 打印编码阶段输出的特征形状
        # print(f"[decode] h_deep形状: {h_deep.shape}, y0_unrelated形状: {y0_unrelated.shape}, y0_related形状: {y0_related.shape}")

        # 直接融合两种特征
        batch_size = y0_unrelated.shape[0]
        combined_feat = torch.cat([y0_unrelated, y0_related], dim=1)
        # print(f"[decode] 组合特征形状: {combined_feat.shape}")

        # 使用预定义的适配层调整融合特征的维度
        try:
            # print(f"[debug] combined_feat维度: {combined_feat.shape}, self.feat_adapter.weight.shape: {self.feat_adapter.weight.shape}")
            fused_feat = self.feat_adapter(combined_feat)
            # print(f"[decode] 适配后融合特征形状: {fused_feat.shape}")

            # 重塑融合特征为特征图
            h_shape = h_deep.shape
            # print(f"[decode] h_deep形状: {h_shape}, channels: {self.h_channels}")
            fused_feat = fused_feat.view(batch_size, self.h_channels, 1, 1)
            # print(f"[decode] 重塑后融合特征形状1: {fused_feat.shape}")
            fused_feat = fused_feat.expand(-1, -1, h_shape[2], h_shape[3])
            # print(f"[decode] 重塑后融合特征形状2: {fused_feat.shape}")
        except Exception as e:
            print(f"[错误] 融合特征处理出错: {e}")
        # print(f"[decode] 重塑后融合特征形状: {fused_feat.shape}")

        # 添加残差连接
        h = h_deep + fused_feat
        # print(f"[decode] 添加残差连接后形状: {h.shape}")

        # 使用上采样层恢复特征图尺寸
        # print(f"[decode] 开始上采样过程")
        h = F.relu(self.upsampler1(h))
        # print(f"[decode] 第一次上采样后形状: {h.shape}")
        h = F.relu(self.upsampler2(h))
        # print(f"[decode] 第二次上采样后形状: {h.shape}")
        h = F.relu(self.upsampler3(h))
        # print(f"[decode] 第三次上采样后形状: {h.shape}")

        # 使用预定义的简化解码器生成最终输出
        # print(f"[decode] 使用简化版解码器生成最终输出")
        h = self.simple_decoder(h)

        # print(f"[decode] 最终输出形状: {h.shape}")
        return h

    def forward(self, x, return_features=False):
        """
        前向传播：编码->解码
        """
        try:
            y0_unrelated, y0_related, h_deep = self.encode(x)
            # print(f"[forward] unrelated形状: {y0_unrelated.shape}, related形状: {y0_related.shape}")
            reconstructed = self.decode(y0_unrelated, y0_related, h_deep)

            if return_features:
                return reconstructed, y0_unrelated, y0_related
            return reconstructed
        except Exception as e:
            print(f"[错误] 前向传播出错: {e}")
            import traceback
            traceback.print_exc()
            raise e


def print_model_structure(model):
    """打印UNetModel的结构，包括每个模块的输入输出通道数"""
    print("\n===== UNetModel结构分析 =====")
    if hasattr(model, 'input_blocks'):
        print("\n输入模块:")
        for i, block in enumerate(model.input_blocks):
            print(f"  输入块 {i}: {block}")

    if hasattr(model, 'middle_block'):
        print("\n中间模块:")
        print(f"  {model.middle_block}")

    if hasattr(model, 'output_blocks'):
        print("\n输出模块:")
        for i, block in enumerate(model.output_blocks):
            print(f"  输出块 {i}: {block}")

    if hasattr(model, 'out'):
        print("\n最终输出层:")
        print(f"  {model.out}")


def train_mrm_idkl(args):
    """
    IDKL-MRM训练主函数 - 实现模态无关和模态相关特征的解耦训练
    """
    device = torch.device(args.device)

    print(f"使用设备: {device}")

    # 记录起始epoch和global_step
    start_epoch = 0
    global_step = 0

    # 初始化wandb（如果开启）
    if args.use_wandb and args.use_wandb.lower() != 'false':
        try:
            wandb.init(
                entity=args.wandb_entity,
                project=args.project_name,
                name=args.exp_name,
                config=vars(args)
            )
            print("成功初始化wandb")
        except ImportError:
            print("未安装wandb包，跳过监控")
            args.use_wandb = 'false'
        except Exception as e:
            print(f"初始化wandb时出错: {e}")
            args.use_wandb = 'false'

    # 数据准备和转换
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 转换到 [-1, 1] 范围，3通道
    ])

    # 从文件夹加载
    dataset = pair_Dataset(
        path_sar="./data/sar",
        path_opt="./data/opt",
        transforms_sar=transform_sar,
        transforms_opt=transform_opt
    )

    # 修改一个扩展的SAR2OptDataset类来跟踪模态信息
    import time

    class ModifiedSAR2OptDataset(SAR2OptDataset):
        def __init__(self, data_dir, transform=None, modality_type='unknown'):
            super().__init__(data_dir, transform)
            self.modality_type = modality_type  # 记录模态类型

        def __getitem__(self, idx):
            # 调用父类的__getitem__获取图像
            image = super().__getitem__(idx)
            # 返回图像，不需要在这里返回模态标签，因为我们会在训练循环中手动创建标签
            return image

    # 加载SAR数据集（模态A）
    print(f"加载SAR数据集自: {args.sar_data_dir}")
    try:
        sar_dataset = ModifiedSAR2OptDataset(
            data_dir=Path(args.sar_data_dir),
            transform=transform,
            modality_type='sar'  # 标记为SAR模态
        )
    except Exception as e:
        print(f"\033[91m错误: 加载SAR数据集失败 - {e}\033[0m")
        return

    # 加载光学数据集（模态B）
    print(f"加载光学数据集自: {args.opt_data_dir}")
    try:
        opt_dataset = ModifiedSAR2OptDataset(
            data_dir=Path(args.opt_data_dir),
            transform=transform,
            modality_type='opt'  # 标记为光学模态
        )
    except Exception as e:
        print(f"\033[91m错误: 加载光学数据集失败 - {e}\033[0m")
        return

    print(f"SAR数据集大小: {len(sar_dataset)} 张图片")
    print(f"OPT数据集大小: {len(opt_dataset)} 张图片")

    if len(sar_dataset) == 0 or len(opt_dataset) == 0:
        raise ValueError(f"未找到有效的图片。请检查数据路径")

    # 可选：减少数据量以加快训练速度
    if args.reduce_data_factor > 1:
        # SAR数据集采样
        original_sar_size = len(sar_dataset)
        sar_indices = list(range(original_sar_size))
        random.shuffle(sar_indices)
        selected_sar_size = original_sar_size // args.reduce_data_factor
        selected_sar_indices = sar_indices[:selected_sar_size]
        sar_dataset = Subset(sar_dataset, selected_sar_indices)

        # 光学数据集采样
        original_opt_size = len(opt_dataset)
        opt_indices = list(range(original_opt_size))
        random.shuffle(opt_indices)
        selected_opt_size = original_opt_size // args.reduce_data_factor
        selected_opt_indices = opt_indices[:selected_opt_size]
        opt_dataset = Subset(opt_dataset, selected_opt_indices)

        print(f"为加快训练，随机选择了 {len(sar_dataset)} 张SAR图片和 {len(opt_dataset)} 张光学图片")

    # 创建数据加载器
    sar_dataloader = DataLoader(
        sar_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    opt_dataloader = DataLoader(
        opt_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 初始化基础UNetModel
    # 确保输入通道数为3（RGB图像）
    base_model = UNetModel(
        image_size=args.image_size,
        in_channels=3,  # 确保输入通道数为3
        model_channels=args.mrm_model_channels,
        out_channels=3,  # 输出通道数也为3
        num_res_blocks=args.mrm_num_res_blocks,
        attention_resolutions=tuple(map(int, args.mrm_attention_resolutions.split(','))),
        channel_mult=tuple(map(int, args.mrm_channel_mult.split(','))),
        use_fp16=args.use_fp16
    ).to(device)

    # 打印模型信息和训练参数
    print("\nUNetModel结构:")
    print_model_structure(base_model)

    # 初始化IDKL扩展模型
    mrm_model = IDKLUNetModel(
        base_model=base_model,
        h_dim=args.feature_dim,
        h_channels=192,
        alpha=0.1
    ).to(device)

    # 初始化模态判别器和混淆器
    # 注：模态判别器用于模态相关特征，混淆器用于模态无关特征（通过梯度反转）
    modality_discriminator = Discrimination(dim=args.feature_dim).to(device)
    modality_confuser = Discrimination(dim=args.feature_dim).to(device)

    # 优化器
    # 为主模型、判别器和混淆器分别创建优化器
    optimizer = optim.Adam(
        mrm_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    optimizer_discriminator = optim.Adam(
        modality_discriminator.parameters(),
        lr=args.learning_rate_disc,
        weight_decay=args.weight_decay
    )

    optimizer_confuser = optim.Adam(
        modality_confuser.parameters(),
        lr=args.learning_rate_disc,
        weight_decay=args.weight_decay
    )

    # 损失函数
    # 重建损失 - L2损失
    criterion_recon = nn.MSELoss()
    # 分类损失 - 交叉熵
    criterion_cls = nn.CrossEntropyLoss()

    # 设置混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"模型将保存到: {args.output_dir}")

    # 恢复训练检查点（如果有）
    if (args.resume.lower() == 'true' or args.auto_resume) and args.ckpt_path:
        if os.path.exists(args.ckpt_path):
            print(f"恢复训练自检查点: {args.ckpt_path}")
            checkpoint = torch.load(args.ckpt_path, map_location=device)
            mrm_model.load_state_dict(checkpoint['model_state_dict'])
            modality_discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            modality_confuser.load_state_dict(checkpoint['confuser_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
            optimizer_confuser.load_state_dict(checkpoint['optimizer_confuser_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            print(f"从第 {start_epoch} 轮和步骤 {global_step} 恢复训练")
        else:
            print(f"未找到检查点: {args.ckpt_path}，从头开始训练")

    # 记录最佳损失值
    best_loss = float('inf')

    # 训练循环
    print("开始IDKL-MRM训练...")
    for epoch in range(start_epoch, args.num_epochs):
        mrm_model.train()
        modality_discriminator.train()
        modality_confuser.train()

        # 跟踪各种损失值
        epoch_recon_loss = 0
        epoch_related_disc_loss = 0
        epoch_unrelated_conf_loss = 0
        epoch_total_loss = 0

        # 使用迭代器来交替处理两种模态的数据
        sar_iter = iter(sar_dataloader)
        opt_iter = iter(opt_dataloader)

        # 取较短的数据集长度作为每个epoch的迭代次数
        num_iterations = min(len(sar_dataloader), len(opt_dataloader))

        # tqdm进度条
        with tqdm(total=num_iterations, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch") as pbar:
            for i in range(num_iterations):
                # 获取SAR和光学数据批次
                try:
                    sar_images = next(sar_iter)
                except StopIteration:
                    sar_iter = iter(sar_dataloader)
                    sar_images = next(sar_iter)

                try:
                    opt_images = next(opt_iter)
                except StopIteration:
                    opt_iter = iter(opt_dataloader)
                    opt_images = next(opt_iter)

                # 将数据移至GPU
                sar_images = sar_images.to(device)
                opt_images = opt_images.to(device)

                # 合并批次用于批量处理
                images = torch.cat([sar_images, opt_images], dim=0)

                # 创建模态标签：0表示SAR，1表示光学
                modality_labels = torch.cat([
                    torch.zeros(sar_images.size(0), dtype=torch.long),
                    torch.ones(opt_images.size(0), dtype=torch.long)
                ]).to(device)

                # 应用掩码
                masked_images, mask_info = mask_func(
                    images,
                    3,  # 通道数
                    args.mask_ratio,
                    args.patch_size,
                    (args.image_size, args.image_size)
                )

                # 前向传播和损失计算
                if args.use_fp16:
                    with torch.cuda.amp.autocast():
                        # 1. 主模型前向传播
                        reconstructed, y0_unrelated, y0_related = mrm_model(masked_images, return_features=True)

                        # 2. 重建损失
                        recon_loss = criterion_recon(reconstructed, images)

                        # 3. 模态相关特征判别损失 - 判别器应该能区分不同模态
                        related_pred = modality_discriminator(y0_related)
                        related_disc_loss = criterion_cls(related_pred, modality_labels)

                        # 4. 模态无关特征混淆损失 - 混淆器应该无法区分不同模态（通过梯度反转）
                        # 使用梯度反转层处理无关特征
                        reversed_unrelated = mrm_model.grl(y0_unrelated, mrm_model.alpha)
                        unrelated_pred = modality_confuser(reversed_unrelated)
                        unrelated_conf_loss = criterion_cls(unrelated_pred, modality_labels)

                        # 总损失 = 重建损失 + λ1*相关判别损失 + λ2*无关混淆损失
                        total_loss = recon_loss + \
                                     args.lambda_related * related_disc_loss + \
                                     args.lambda_unrelated * unrelated_conf_loss
                else:
                    # 非混合精度的前向传播与损失计算
                    reconstructed, y0_unrelated, y0_related = mrm_model(masked_images, return_features=True)

                    recon_loss = criterion_recon(reconstructed, images)

                    related_pred = modality_discriminator(y0_related)
                    related_disc_loss = criterion_cls(related_pred, modality_labels)

                    # 使用已经定义的rev_layer而不是grl
                    reversed_unrelated = mrm_model.rev_layer(y0_unrelated)
                    unrelated_pred = modality_confuser(reversed_unrelated)
                    unrelated_conf_loss = criterion_cls(unrelated_pred, modality_labels)

                    total_loss = recon_loss + \
                                 args.lambda_related * related_disc_loss + \
                                 args.lambda_unrelated * unrelated_conf_loss

                # 优化主模型
                optimizer.zero_grad()

                if args.use_fp16:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                # 分别训练判别器和混淆器
                # 1. 训练模态判别器（用于模态相关特征）
                optimizer_discriminator.zero_grad()
                with torch.no_grad():
                    _, _, y0_related = mrm_model(masked_images, return_features=True)
                related_pred = modality_discriminator(y0_related.detach())
                related_disc_loss_only = criterion_cls(related_pred, modality_labels)

                if args.use_fp16:
                    scaler.scale(related_disc_loss_only).backward()
                    scaler.step(optimizer_discriminator)
                    scaler.update()
                else:
                    related_disc_loss_only.backward()
                    optimizer_discriminator.step()

                # 2. 训练模态混淆器（用于模态无关特征）
                optimizer_confuser.zero_grad()
                with torch.no_grad():
                    _, y0_unrelated, _ = mrm_model(masked_images, return_features=True)
                unrelated_pred = modality_confuser(y0_unrelated.detach())
                unrelated_conf_loss_only = criterion_cls(unrelated_pred, modality_labels)

                if args.use_fp16:
                    scaler.scale(unrelated_conf_loss_only).backward()
                    scaler.step(optimizer_confuser)
                    scaler.update()
                else:
                    unrelated_conf_loss_only.backward()
                    optimizer_confuser.step()

                # 更新损失统计
                epoch_recon_loss += recon_loss.item()
                epoch_related_disc_loss += related_disc_loss.item()
                epoch_unrelated_conf_loss += unrelated_conf_loss.item()
                epoch_total_loss += total_loss.item()
                global_step += 1

                # 更新进度条
                pbar.set_postfix({
                    "total_loss": f"{total_loss.item():.4f}",
                    "recon_loss": f"{recon_loss.item():.4f}",
                    "disc_loss": f"{related_disc_loss.item():.4f}",
                    "conf_loss": f"{unrelated_conf_loss.item():.4f}"
                })
                pbar.update(1)

                # 记录到wandb
                if args.use_wandb and args.use_wandb.lower() != 'false':
                    wandb.log({
                        "train_recon_loss": recon_loss.item(),
                        "train_related_disc_loss": related_disc_loss.item(),
                        "train_unrelated_conf_loss": unrelated_conf_loss.item(),
                        "train_total_loss": total_loss.item(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    })

                # 定期保存检查点
                if global_step % args.save_interval == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"idkl_mrm_model_step{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': mrm_model.state_dict(),
                        'discriminator_state_dict': modality_discriminator.state_dict(),
                        'confuser_state_dict': modality_confuser.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
                        'optimizer_confuser_state_dict': optimizer_confuser.state_dict(),
                        'loss': total_loss.item(),
                    }, checkpoint_path)
                    print(f"保存检查点到: {checkpoint_path}")

                    # 保存一些重建结果用于可视化
                    if args.use_wandb and args.use_wandb.lower() != 'false':
                        with torch.no_grad():
                            # 选择一批中的前4张图片进行可视化
                            vis_images = images[:4].cpu().numpy()
                            vis_masked = masked_images[:4].cpu().numpy()
                            vis_recon = reconstructed[:4].cpu().numpy()

                            # 转换回[0,1]范围用于显示
                            vis_images = (vis_images + 1) / 2
                            vis_masked = (vis_masked + 1) / 2
                            vis_recon = (vis_recon + 1) / 2

                            # 将图像剪裁到[0,1]范围
                            vis_images = np.clip(vis_images, 0, 1)
                            vis_masked = np.clip(vis_masked, 0, 1)
                            vis_recon = np.clip(vis_recon, 0, 1)

                            # 记录到wandb
                            wandb.log({
                                "original_images": [wandb.Image(img.transpose(1, 2, 0)) for img in vis_images],
                                "masked_images": [wandb.Image(img.transpose(1, 2, 0)) for img in vis_masked],
                                "reconstructed_images": [wandb.Image(img.transpose(1, 2, 0)) for img in vis_recon],
                            })

        # 每个epoch结束后计算平均损失
        avg_recon_loss = epoch_recon_loss / num_iterations
        avg_related_disc_loss = epoch_related_disc_loss / num_iterations
        avg_unrelated_conf_loss = epoch_unrelated_conf_loss / num_iterations
        avg_total_loss = epoch_total_loss / num_iterations

        print(f"Epoch {epoch + 1}/{args.num_epochs}, "
              f"平均重建损失: {avg_recon_loss:.4f}, "
              f"平均相关判别损失: {avg_related_disc_loss:.4f}, "
              f"平均无关混淆损失: {avg_unrelated_conf_loss:.4f}, "
              f"平均总损失: {avg_total_loss:.4f}")

        # 保存每个epoch的检查点
        checkpoint_path = os.path.join(args.output_dir, f"idkl_mrm_model_epoch{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': mrm_model.state_dict(),
            'discriminator_state_dict': modality_discriminator.state_dict(),
            'confuser_state_dict': modality_confuser.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
            'optimizer_confuser_state_dict': optimizer_confuser.state_dict(),
            'loss': avg_total_loss,
        }, checkpoint_path)
        print(f"保存第 {epoch + 1} 轮检查点到: {checkpoint_path}")

        # 保存最佳模型
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_model_path = os.path.join(args.output_dir, "idkl_mrm_model_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': mrm_model.state_dict(),
                'discriminator_state_dict': modality_discriminator.state_dict(),
                'confuser_state_dict': modality_confuser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
                'optimizer_confuser_state_dict': optimizer_confuser.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"保存最佳模型到: {best_model_path}, 总损失: {best_loss:.4f}")

    print("IDKL-MRM训练完成！")


if __name__ == "__main__":
    # 对于 Linux 或 macOS, 如果需要代理，请设置环境变量
    proxy_url = "http://127.0.0.1:7890"  # 这是一个示例，请使用你自己的代理地址
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url

    parser = argparse.ArgumentParser(description="训练SAR到光学的IDKL-MRM模型（模态无关与相关特征解耦）")

    # 基本参数
    parser.add_argument("--sar_data_dir", type=str, default="/home/wushixuan/yujun/data/rsdiffusion/sar2opt/trainA",
                        help="SAR数据目录路径")
    parser.add_argument("--opt_data_dir", type=str, default="/home/wushixuan/yujun/data/rsdiffusion/sar2opt/trainB",
                        help="光学数据目录路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/mrm_idkl_sar2opt", help="模型输出目录")
    parser.add_argument("--image_size", type=int, default=256, help="输入图像大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作线程数")

    # 恢复训练相关参数
    parser.add_argument("--resume", type=str, default="false", help="是否从检查点恢复训练 (true/false)")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="要加载的检查点路径，为空则尝试自动恢复最新检查点")
    parser.add_argument("--auto_resume", action='store_true', help="自动从最新检查点恢复训练")

    # MRM模型参数 (UNetModel)
    parser.add_argument("--mrm_in_channels", type=int, default=3,
                        help="MRM输入图像通道数（对于RGB图像为3）")
    parser.add_argument("--mrm_out_channels", type=int, default=3, help="MRM输出图像通道数（与输入一致，为3）")
    parser.add_argument("--mrm_model_channels", type=int, default=96, help="MRM模型基础通道数")
    parser.add_argument("--mrm_num_res_blocks", type=int, default=1, help="MRM每个下采样的残差块数")
    parser.add_argument("--mrm_attention_resolutions", type=str, default="32,16,8", help="MRM注意力分辨率，逗号分隔")
    parser.add_argument("--mrm_channel_mult", type=str, default="1,1,2,2", help="MRM通道乘数，逗号分隔")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="随机掩码比例")
    parser.add_argument("--patch_size", type=str, default="16,16",
                        help="图像补丁大小，逗号分隔")

    # IDKL特定参数
    parser.add_argument("--feature_dim", type=int, default=192, help="特征维度大小")
    parser.add_argument("--lambda_related", type=float, default=0.1, help="模态相关特征判别损失权重")
    parser.add_argument("--lambda_unrelated", type=float, default=0.1, help="模态无关特征混淆损失权重")
    parser.add_argument("--reduce_data_factor", type=int, default=1, help="数据集缩减因子，用于加快训练")

    # 训练超参数
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=12, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="主模型学习率")
    parser.add_argument("--learning_rate_disc", type=float, default=5e-5, help="判别器和混淆器学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam优化器的权重衰减率")
    parser.add_argument("--use_fp16", action='store_true', help="使用混合精度训练")
    parser.add_argument("--save_interval", type=int, default=1000, help="保存检查点间隔步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")

    # wandb监控设置
    parser.add_argument("--use_wandb", type=str, default="false", help="是否启用wandb监控")
    parser.add_argument("--project_name", type=str, default="SAR2Opt-IDKL-MRM-Training", help="wandb项目名称")
    parser.add_argument("--exp_name", type=str, default="idkl-mrm-sar2opt-exp", help="wandb实验名称")
    parser.add_argument("--wandb_entity", type=str, default="wushixuan238-anhui-university", help="wandb实体名称")

    args = parser.parse_args()

    # 转换patch_size为元组
    args.patch_size = tuple(map(int, args.patch_size.split(',')))

    # 设置随机种子
    set_seed(args.seed)

    # 打印主要参数
    print("=" * 50)
    print(f"SAR2Opt IDKL-MRM训练参数设置:")
    print(f"  SAR数据路径: {args.sar_data_dir}")
    print(f"  光学数据路径: {args.opt_data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  主模型学习率: {args.learning_rate}")
    print(f"  判别器学习率: {args.learning_rate_disc}")
    print(f"  设备: {args.device}")
    print(f"  图像大小: {args.image_size}")
    print(f"  MRM输入/输出通道: {args.mrm_in_channels}/{args.mrm_out_channels}")
    print(f"  掩码比例: {args.mask_ratio}")
    print(f"  补丁大小: {args.patch_size}")
    print(f"  特征维度: {args.feature_dim}")
    print(f"  模态相关特征损失权重: {args.lambda_related}")
    print(f"  模态无关特征损失权重: {args.lambda_unrelated}")
    print(f"  使用FP16: {args.use_fp16}")
    print("=" * 50)

    # 开始训练
    train_mrm_idkl(args)
