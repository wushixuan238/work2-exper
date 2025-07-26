import copy
import functools
import os
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

# 检查是否使用分布式训练
USE_DISTRIBUTED = os.environ.get("USE_DISTRIBUTED", "0") == "1"
import blobfile as bf
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import wandb

class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            use_wandb=False,  # 添加wandb支持参数
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        # wandb配置
        self.use_wandb = use_wandb

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        # 在单机模式下或无可用的CUDA，直接使用模型而不包装为DDP
        if not USE_DISTRIBUTED or not th.cuda.is_available():
            self.use_ddp = False
            self.ddp_model = self.model
            if dist.get_world_size() > 1 and th.cuda.is_available():
                logger.warn(
                    "分布式训练设置为禁用，但世界大小 > 1。"
                    "检查您的USE_DISTRIBUTED设置。"
                )
        # 分布式模式且有CUDA可用时使用DDP
        elif th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "分布式训练需要CUDA支持。"
                    "梦境将不会正确同步！"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            loss_dict = self.run_step(batch, cond)
            if self.use_wandb:
                wandb.log(loss_dict)
            if (self.step + self.resume_step) % self.log_interval == 0:
                logger.dumpkvs()
            if (self.step + self.resume_step) % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        loss_dict = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return loss_dict

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        loss_dict = {}
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            micro_sar, micro_opt = th.split(micro, 3, dim=1)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_opt,
                micro_sar,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            batch_losses = log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            loss_dict.update(batch_losses)
            self.mp_trainer.backward(loss)
            
        return loss_dict

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # 移除分布式训练相关代码
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # 移除分布式训练相关代码
        with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses, to_wandb=False):
    """记录损失值到logger，现在包括wandb"""
    loss_dict = {}
    # 记录平均损失
    for key, values in losses.items():
        mean_loss = values.mean().item()
        logger.logkv_mean(key, mean_loss)
        loss_dict[key] = mean_loss
        # 记录四分位数
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            quartile_name = f"{key}_q{quartile}"
            logger.logkv_mean(quartile_name, sub_loss)
            loss_dict[quartile_name] = sub_loss
    
    # 返回损失字典，供WandbTrainLoop使用
    return loss_dict


class WandbTrainLoop(TrainLoop):
    """
    支持Weights & Biases的训练循环类，用于详细记录训练损失和指标
    """
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_wandb=True,
        wandb_project="conditional-diffusion",
        wandb_run_name=None,
        **kwargs
    ):
        # 确保wandb已初始化
        if use_wandb and wandb.run is None:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "lr_anneal_steps": lr_anneal_steps,
                    "ema_rate": ema_rate,
                    "use_fp16": use_fp16,
                    **kwargs
                }
            )
        
        super().__init__(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=batch_size,
            microbatch=microbatch,
            lr=lr,
            ema_rate=ema_rate,
            log_interval=log_interval,
            save_interval=save_interval,
            resume_checkpoint=resume_checkpoint,
            use_fp16=use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=weight_decay,
            lr_anneal_steps=lr_anneal_steps,
            use_wandb=use_wandb
        )
        self.current_losses = {}
    
    def forward_backward(self, batch, cond):
        """重写forward_backward以捕获详细的损失值"""
        self.mp_trainer.zero_grad()
        self.current_losses = {}
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            # SAR->Optical模型特定处理
            if micro.shape[1] == 6:  # 假设是连接的SAR和光学图像
                micro_sar, micro_opt = th.split(micro, 3, dim=1)
            else:
                micro_sar, micro_opt = micro, micro  # 如果没有拆分，就假设是同一图像

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_opt,
                micro_sar,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            
            # 保存所有类型的损失用于wandb记录
            weighted_losses = {k: (v * weights).mean().item() for k, v in losses.items()}
            self.current_losses.update(weighted_losses)
            
            # 标准的损失记录
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            
            self.mp_trainer.backward(loss)
        
        return self.current_losses
    
    def run_step(self, batch, cond):
        """重写run_step以在每一步后记录详细的损失到wandb"""
        # 调用自定义forward_backward获取损失
        loss_dict = self.forward_backward(batch, cond)
        
        # 优化步骤保持不变
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        
        # 添加额外指标到损失字典
        if self.lr_anneal_steps:
            frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
            lr = self.lr * (1 - frac_done)
            loss_dict["learning_rate"] = lr
        else:
            loss_dict["learning_rate"] = self.lr
        
        loss_dict["step"] = self.step + self.resume_step
        loss_dict["samples"] = (self.step + self.resume_step + 1) * self.global_batch
        
        # 如果使用wandb，记录所有损失
        if self.use_wandb:
            wandb.log(loss_dict, step=self.step + self.resume_step)
        
        return loss_dict
