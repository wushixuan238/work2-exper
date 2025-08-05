import datetime
import logging
import os
import random
import re
import shutil
from collections import namedtuple
from typing import List, Optional
from huggingface_hub.utils import enable_progress_bars
import torch.nn as nn
from examples.training.utils.image_datasets import load_data
from lbm.models.conditional_mdn import ConditionalMDN

from cross_conditioned_unet_model.guided_diffusion.unet_rdm import UNetModel

enable_progress_bars()
import braceexpand
import fire
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionXLPipeline, StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import InterpolationMode

from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger
from lbm.trainer.utils import StateDictAdapter

# Define a simple namedtuple for the config
ConditionalMDNEmbedderConfig = namedtuple(
    "ConditionalMDNEmbedderConfig",
    ["c_mdn_model_name", "feature_dim", "condition_dim"]
)


# Add a to_dict() method to it
def to_dict(self):
    return self._asdict()


ConditionalMDNEmbedderConfig.to_dict = to_dict


class ConditionalMDNEmbedder(torch.nn.Module):
    def __init__(self, c_mdn, feature_dim, condition_dim):
        super().__init__()
        self.c_mdn = c_mdn
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.config = ConditionalMDNEmbedderConfig(
            c_mdn_model_name=c_mdn.__class__.__name__,
            feature_dim=feature_dim,
            condition_dim=condition_dim,
        )
        self.projection_layer = nn.Linear(feature_dim, 1024)
        # 修正：添加 input_key 属性以满足框架接口要求
        self.input_key = None

        # 修正：添加 ucg_rate 属性以满足框架接口要求
        self.ucg_rate = 0.0

        # 冻结模型
        self.c_mdn.eval()
        for param in self.c_mdn.parameters():
            param.requires_grad = False

    # 修正：添加 on_fit_start 方法以满足框架要求
    def on_fit_start(self, device=None, *args, **kwargs):
        pass

    def forward(self, batch, **kwargs):
        # LBM 框架会传入整个 batch，但我们只需要获取设备的batch_size
        dtype = self.c_mdn.input_proj.weight.dtype  # 或者其他任意一个参数的dtype
        if self.training:
            bsz = batch["target_image"].shape[0]
            device = batch["target_image"].device

            # 1. 生成一个随机噪声，作为MDN的输入
            # 这里的形状需要与 ConditionalMDN 的 'noisy_style_tokens' 期望的形状匹配
            # 我们需要知道这个形状。根据你的 ConditionalMDN 的定义，它期望的输入形状是 (B, N, D_feat)

            # 这里我用一个假设的形状 (16, 768)

            # 假设 N 是 patches 数量，D 是特征维度
            num_patches = (batch["source_image"].shape[2] // 16) * (batch["source_image"].shape[3] // 16)
        else:
            bsz = batch["source_image"].shape[0]
            device = batch["source_image"].device
            num_patches = (batch["source_image"].shape[2] // 16) * (batch["source_image"].shape[3] // 16)

        # 'noisy_style_tokens' 的形状: (B, N, feature_dim)
        noisy_style_tokens = torch.randn(bsz, num_patches, self.feature_dim, device=device,
                                         dtype=dtype)

        # 'content_condition_tokens' 的形状: (B, N, condition_dim)
        content_condition_tokens = torch.randn(bsz, num_patches, self.condition_dim, device=device,
                                               dtype=dtype)

        # 2. 采样时间步
        total_timesteps = 1000
        timesteps = torch.randint(0, total_timesteps, (bsz,), device=device).long()

        # 3. 使用 ConditionalMDN 生成分布
        noise_pred = self.c_mdn(
            noisy_style_tokens=noisy_style_tokens,
            time_steps=timesteps,
            content_condition_tokens=content_condition_tokens  # 这里我们直接使用随机噪声作为条件
        )
        # print(f"DEBUG: ConditionalMDN output shape: {noise_pred.shape}")
        # 返回一个字典，LBM框架会将其作为条件
        projected_noise_pred = self.projection_layer(noise_pred)

        return {"conditional_mdn_output": projected_noise_pred}


def get_model(
        backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_num_channels: int = 4,
        unet_input_channels: int = 4,
        timestep_sampling: str = "log_normal",
        selected_timesteps: Optional[List[float]] = None,
        prob: Optional[List[float]] = None,
        conditioning_images_keys: Optional[List[str]] = [],
        conditioning_masks_keys: Optional[List[str]] = [],
        source_key: str = "source_image",
        target_key: str = "source_image_paste",
        mask_key: str = "mask",
        bridge_noise_sigma: float = 0.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        pixel_loss_type: str = "lpips",
        latent_loss_type: str = "l2",
        latent_loss_weight: float = 1.0,
        pixel_loss_weight: float = 0.0,
        feature_dim=None,  # 添加这行
        condition_dim=None,  # 添加这行
        conditional_mdn_ckpt_path="/home/wushixuan/桌面/07/work2-exper/wushixuan/checkpoints_stage3_cmdn/c_mdn_best.pt",
        image_size: int = 512,  # 添加 image_size 参数
        patch_size: int = 16,  # 添加 patch_size 参数

):
    conditioners = []

    # --- 开始修改 ---
    print("正在初始化自定义 UNetModel...")

    # 假设 VAE 的下采样倍数是 8 (例如，512x512 -> 64x64)
    # 您的 `UNetModel` 似乎是为 token 序列设计的，但 LBM 框架需要一个 UNet
    # 作用于 VAE 的潜在图像。这里我们直接配置一个标准的 UNet。
    # 潜在空间的尺寸是 image_size // 8
    latent_spatial_dim = image_size // 8

    # 实例化您的自定义 UNetModel。
    # UNet 的输入/输出通道数应与 VAE 潜在空间的通道数（vae_num_channels）一致
    # image_size 应该设为潜在空间的尺寸
    denoiser = UNetModel(
        image_size=latent_spatial_dim,
        in_channels=vae_num_channels,
        model_channels=320,  # 一个常见的 UNet 基础通道数
        out_channels=vae_num_channels,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        channel_mult=[1, 2, 4, 4],
        use_checkpoint=True,
        use_scale_shift_norm=False,  # SD 2.1 风格的 UNet 不使用此项
        use_fp16=True if torch.cuda.is_available() else False,
        num_heads=8,
    ).to(torch.bfloat16)

    print("自定义 UNetModel 创建成功。")

    # # Load pretrained model as base
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     backbone_signature,
    #     torch_dtype=torch.bfloat16,
    # )
    #
    # denoiser = pipe.unet
    # # 如果你的任务需要改变输入通道数（比如拼接了法线贴图）
    # # 你需要手动替换它的第一个卷积层
    # # 假设你的 unet_input_channels > 4
    # if denoiser.config.in_channels != unet_input_channels:
    #     original_conv_in = denoiser.conv_in
    #     denoiser.config.in_channels = unet_input_channels  # 更新配置
    #
    #     # 创建一个新的输入卷积层
    #     new_conv_in = torch.nn.Conv2d(
    #         in_channels=unet_input_channels,
    #         out_channels=original_conv_in.out_channels,
    #         kernel_size=original_conv_in.kernel_size,
    #         stride=original_conv_in.stride,
    #         padding=original_conv_in.padding,
    #     )
    #
    #     # 将原始权重复制到新层的前4个通道
    #     with torch.no_grad():
    #         new_conv_in.weight[:, :4, :, :] = original_conv_in.weight
    #         # 新增通道的权重默认为0
    #         new_conv_in.weight[:, 4:, :, :].zero_()
    #         new_conv_in.bias = original_conv_in.bias
    #
    #     denoiser.conv_in = new_conv_in
    #
    # ### MMMDiT ###
    # # Get Architecture
    #
    # state_dict = pipe.unet.state_dict()

    # del state_dict["add_embedding.linear_1.weight"]
    # del state_dict["add_embedding.linear_1.bias"]
    # del state_dict["add_embedding.linear_2.weight"]
    # del state_dict["add_embedding.linear_2.bias"]

    # Adapt the shapes
    # state_dict_adapter = StateDictAdapter()
    # state_dict = state_dict_adapter(
    #     model_state_dict=denoiser.state_dict(),
    #     checkpoint_state_dict=state_dict,
    #     regex_keys=[
    #         r"class_embedding.linear_\d+.(weight|bias)",
    #         r"conv_in.weight",
    #         r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
    #         r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
    #     ],
    #     strategy="zeros",
    # )

    # denoiser.load_state_dict(state_dict, strict=True)
    #
    # del pipe

    # if conditioning_images_keys != [] or conditioning_masks_keys != []:
    #     latents_concat_embedder_config = LatentsConcatEmbedderConfig(
    #         image_keys=conditioning_images_keys,
    #         mask_keys=conditioning_masks_keys,
    #     )
    #     latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
    #     latent_concat_embedder.freeze()
    #     conditioners.append(latent_concat_embedder)

    # 1. 加载并冻结 ConditionalMDN
    conditional_mdn = ConditionalMDN(
        feature_dim=feature_dim,
        condition_dim=condition_dim
    )
    print(f"Loading ConditionalMDN checkpoint from: {conditional_mdn_ckpt_path}")

    # 确保文件存在
    if not os.path.exists(conditional_mdn_ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {conditional_mdn_ckpt_path}")

    conditional_mdn.load_state_dict(torch.load(conditional_mdn_ckpt_path, map_location="cpu"))
    print("ConditionalMDN模型加载成功")
    print(conditional_mdn)

    # 修正：移除对不存在的 freeze() 方法的调用，直接使用 PyTorch 标准方法冻结参数
    for param in conditional_mdn.parameters():
        param.requires_grad = False

    print("ConditionalMDN模型参数已冻结。")

    # 2. 实例化你的自定义 Embedder
    conditional_mdn_embedder = ConditionalMDNEmbedder(
        c_mdn=conditional_mdn,
        feature_dim=feature_dim,
        condition_dim=condition_dim
    )

    # 3. 将其包装成 lbm 框架所需的 ConditionerWrapper
    conditioner = ConditionerWrapper(
        conditioners=[conditional_mdn_embedder],
    )

    # # Wrap conditioners and set to device
    # conditioner = ConditionerWrapper(
    #     conditioners=conditioners,
    # )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch.bfloat16)

    # LBM Config
    config = LBMConfig(
        ucg_keys=None,
        source_key=source_key,  # 将会是 "source_image"
        target_key=target_key,  # 将会是 "target_image"
        mask_key=mask_key,  # 将会是 None，因为我们没有掩码
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    # LBM Model
    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)

    return model


def get_filter_mappers():
    filters_mappers = [
        KeyFilter(KeyFilterConfig(keys=["jpg", "normal_aligned.png", "mask.png"])),
        MapperWrapper(
            [
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            "jpg": "image",
                            "normal_aligned.png": "normal",
                            "mask.png": "mask",
                        }
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="image",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (480, 640),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="normal",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (480, 640),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="mask",
                        transforms=["ToTensor", "Resize", "Normalize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (480, 640),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                            {"mean": 0.0, "std": 1.0},
                        ],
                    )
                ),
                RescaleMapper(RescaleMapperConfig(key="image")),
                RescaleMapper(RescaleMapperConfig(key="normal")),
            ],
        ),
    ]

    return filters_mappers


def get_data_module(
        train_shards: List[str],
        validation_shards: List[str],
        batch_size: int,
):
    # TRAIN
    train_filters_mappers = get_filter_mappers()

    # unbrace urls
    train_shards_path_or_urls_unbraced = []
    for train_shards_path_or_url in train_shards:
        train_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(train_shards_path_or_url)
        )

    # shuffle shards
    random.shuffle(train_shards_path_or_urls_unbraced)

    # data config
    data_config = DataModuleConfig(
        shards_path_or_urls=train_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=20,
        shuffle_before_split_by_workers_buffer_size=20,
        shuffle_before_filter_mappers_buffer_size=20,
        shuffle_after_filter_mappers_buffer_size=20,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    train_data_config = data_config

    # VALIDATION
    validation_filters_mappers = get_filter_mappers()

    # unbrace urls
    validation_shards_path_or_urls_unbraced = []
    for validation_shards_path_or_url in validation_shards:
        validation_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(validation_shards_path_or_url)
        )

    data_config = DataModuleConfig(
        shards_path_or_urls=validation_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=10,
        shuffle_before_split_by_workers_buffer_size=10,
        shuffle_before_filter_mappers_buffer_size=10,
        shuffle_after_filter_mappers_buffer_size=10,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    validation_data_config = data_config

    # data module
    data_module = DataModule(
        train_config=train_data_config,
        train_filters_mappers=train_filters_mappers,
        eval_config=validation_data_config,
        eval_filters_mappers=validation_filters_mappers,
    )

    return data_module


def main(
        # --- 为你的数据加载器添加这些新参数 ---
        conditional_mdn_ckpt_path: str,
        feature_dim: int,
        condition_dim: int,
        data_dir_sar: str,
        data_dir_opt: str,
        image_size: int = 512,
        train_shards: List[str] = ["pipe:cat path/to/train/shards"],
        validation_shards: List[str] = ["pipe:cat path/to/validation/shards"],
        # backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
        backbone_signature: str = "/home/wushixuan/yujun/PycharmProjects/SD/DiffDIS/pretrained_weights/stable-diffusion-2-1",
        vae_num_channels: int = 4,
        unet_input_channels: int = 4,
        # source_key: str = "image",
        # target_key: str = "normal",
        # mask_key: str = "mask",
        source_key: str = "source_image",
        target_key: str = "target_image",
        mask_key: Optional[str] = None,  # 将默认值设为 None,
        wandb_project: str = "lbm-surface",
        batch_size: int = 8,
        num_steps: List[int] = [1, 2, 4],
        learning_rate: float = 5e-5,
        learning_rate_scheduler: str = None,
        learning_rate_scheduler_kwargs: dict = {},
        optimizer: str = "AdamW",
        optimizer_kwargs: dict = {},
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        pixel_loss_type: str = "lpips",
        latent_loss_type: str = "l2",
        latent_loss_weight: float = 1.0,
        pixel_loss_weight: float = 0.0,
        selected_timesteps: List[float] = None,
        prob: List[float] = None,
        conditioning_images_keys: Optional[List[str]] = [],
        conditioning_masks_keys: Optional[List[str]] = [],
        config_yaml: dict = None,
        save_ckpt_path: str = "./checkpoints",
        log_interval: int = 100,
        resume_from_checkpoint: bool = True,
        max_epochs: int = 100,
        bridge_noise_sigma: float = 0.005,
        save_interval: int = 1000,
        path_config: str = None,
):
    print(">>> 1. Entering main function...")
    model = get_model(
        backbone_signature=backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
        bridge_noise_sigma=bridge_noise_sigma,
        feature_dim=feature_dim,  # 添加这行
        condition_dim=condition_dim,  # 添加这行
    )
    print(">>> 2. Model has been loaded successfully.")
    # --- 移除旧的数据模块 ---
    # data_module = get_data_module(...)
    # data_module = get_data_module(
    #     train_shards=train_shards,
    #     validation_shards=validation_shards,
    #     batch_size=batch_size,
    # )
    # print(">>> 3. Data module has been created.")

    # --- 使用你的数据加载器 ---
    # 注意：你的 load_data 函数没有划分训练集和验证集。
    # 暂时，我们让它们使用同一个加载器，但在正式训练时你应该修正这一点。
    train_loader = load_data(
        data_dir_sar=data_dir_sar,
        data_dir_opt=data_dir_opt,
        batch_size=batch_size,
        image_size=image_size
    )
    val_loader = train_loader  # 临时措施：用同样的数据进行验证
    print(">>> 3. 数据加载器已创建。")

    train_parameters = ["denoiser.*"]

    # Training Config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["image", "normal", "mask"],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": num_steps,
        },
    )
    if (
            os.path.exists(save_ckpt_path)
            and resume_from_checkpoint
            and "last.ckpt" in os.listdir(save_ckpt_path)
    ):
        start_ckpt = f"{save_ckpt_path}/last.ckpt"
        print(f"Resuming from checkpoint: {start_ckpt}")

    else:
        start_ckpt = None

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )

    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "vae": model.vae.config.to_dict(),
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )

    training_signature = (
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "-LBM-Surface-SingleGPU"
    )
    dir_path = f"{save_ckpt_path}/logs/{training_signature}"
    os.makedirs(dir_path, exist_ok=True)
    if path_config is not None:
        shutil.copy(path_config, f"{save_ckpt_path}/config.yaml")
    run_name = training_signature

    # Ignore parameters unused during training
    ignore_states = []
    for name, param in pipeline.model.named_parameters():
        ignore = True
        for regex in ["denoiser."]:
            pattern = re.compile(regex)
            if re.match(pattern, name):
                ignore = False
        if ignore:
            ignore_states.append(param)

    # FSDP Strategy
    strategy = FSDPStrategy(
        auto_wrap_policy=ModuleWrapPolicy(
            [
                UNet2DConditionModel,
                BasicTransformerBlock,
                ResnetBlock2D,
                torch.nn.Conv2d,
            ]
        ),
        activation_checkpointing_policy=ModuleWrapPolicy(
            [
                BasicTransformerBlock,
                ResnetBlock2D,
            ]
        ),
        sharding_strategy="SHARD_GRAD_OP",
        ignored_states=ignore_states,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,  # 只使用1个GPU
        strategy=strategy,
        default_root_dir="logs",
        logger=loggers.WandbLogger(
            project=wandb_project, offline=True, name=run_name, save_dir=save_ckpt_path
        ),
        callbacks=[
            WandbSampleLogger(log_batch_freq=log_interval),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                every_n_train_steps=save_interval,
                save_last=True,
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=2,
        # val_check_interval=1000,
        val_check_interval=350,  # <--- 改成一个小于 362 的值
        max_epochs=max_epochs,
    )
    print(">>> 4. Trainer has been initialized. Starting fit...")

    # 直接传递加载器，而不是 data_module
    trainer.fit(
        pipeline,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=start_ckpt
    )
    print(">>> 5. Training finished.")


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    # --path_config ./config/surface.yaml
    import os

    proxy_url = "http://127.0.0.1:7890"  # 这是一个示例，请使用你自己的
    # 设置 HTTP 和 HTTPS 代理环境变量
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url

    os.environ[" HF_HUB_ENABLE_PROGRESS_BARS"] = "1"

    os.environ["WANDB_MODE"] = "disabled"
    fire.Fire(main_from_config)
