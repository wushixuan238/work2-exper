import numpy as np
import torch 
import torch.nn as nn 
from utils.data_utils_brats23_no_norm import get_train_val_dataset
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from einops import rearrange
from utils.metrics import compute_psnr, compute_ssim
import random
import os 
set_determinism(42)
import os
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

from guided_diffusion.unet import UNetModel
 

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
data_dir = "./data/fullres/train"
fold = 0

# logdir = f"./logs_brats23_norm01_filter_data/rdm_unet_240_ep20_sample10"
logdir = f"./logs_brats23_norm01_filter_data/rdm_unet_240_ep20_sample50_test"
env = "pytorch"
model_save_path = os.path.join(logdir, "model")
max_epoch = 20
batch_size = 24
val_every = 1
num_gpus = 1
device = "cuda:0"
image_size = 224


class Pix2PixTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        from guided_diffusion.unet_rdm import UNetModel

        from rcg.rdm.util import instantiate_from_config
        from omegaconf import OmegaConf
        from rcg.rdm.models.diffusion.ddim import DDIMSampler

        config = OmegaConf.load("/home/xingzhaohu/image_synthesis/config/rdm/mocov3vitb_simplemlp_l12_w1536.yaml")

        rdm_model = instantiate_from_config(config.model)
        rdm_model.to(self.device)

        self.sampler = DDIMSampler(model=rdm_model)

        self.rdm_model = rdm_model

        self.model = UNetModel(image_size=240, in_channels=2,
                  model_channels=96, out_channels=2, 
                  num_res_blocks=1, attention_resolutions=[32,16,8],
                  channel_mult=[1, 1, 2, 2])

        self.best_mean_dice = 0.0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=3e-5, eps=1e-8)
        self.scheduler_type = "poly"
        self.loss_func = nn.L1Loss()
        self.mse = nn.MSELoss()
    
    def filte_sd(self, weight_path, strict=True):
        sd = torch.load(weight_path, map_location="cpu")
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        return new_sd    

    def training_step(self, batch):

        image_t1n, image_t1c, image_t2w, image_t2f = batch["t1n"], batch["t1c"], batch["t2w"], batch["t2f"]
        inputs = torch.cat([image_t1n, image_t2w], dim=1)

        # print(inputs.min(), inputs.max())
        outputs = torch.cat([image_t1c, image_t2f], dim=1)        
        sampled_rep, _ = self.sampler.sample(50, conditioning=None, batch_size=inputs.shape[0],
                                                                        shape=(192, 1, 1),
                                                                        eta=1.0, verbose=False)
        sampled_rep = sampled_rep.squeeze(-1).squeeze(-1)

        pred = self.model(inputs, sampled_rep)
       
        loss = self.mse(pred, outputs)
        
        self.log("loss", loss, step=self.global_step)

        return loss 
    
    def cal_metric(self, pred, gt):

        # pred = pred * 0.5 + 0.5
        pred = pred.clamp(0, 1)
    
        # gt = gt * 0.5 + 0.5
        gt = gt.clamp(0, 1)

        pred = pred.cpu().numpy()[0]
        gt = gt.cpu().numpy()[0]
        
        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)

        # if psnr > 1000:
        #     psnr = 40

        mae = np.mean(np.abs(pred - gt))
        return psnr, ssim, mae 
    
    def validation_step(self, batch):

        image_t1n, image_t1c, image_t2w, image_t2f = batch["t1n"], batch["t1c"], batch["t2w"], batch["t2f"]
        inputs = torch.cat([image_t1n, image_t2w], dim=1)

        outputs = torch.cat([image_t1c, image_t2f], dim=1)
        
        sampled_rep, _ = self.sampler.sample(50, conditioning=None, batch_size=1,
                                                                        shape=(192, 1, 1),
                                                                        eta=1.0, verbose=False)
        sampled_rep = sampled_rep.squeeze(-1).squeeze(-1)

        pred = self.model(inputs, sampled_rep)
        
        psnr_1, ssim_1, mae_1 = self.cal_metric(pred[:, 0:1], outputs[:, 0:1])
        psnr_2, ssim_2, mae_2 = self.cal_metric(pred[:, 1:2], outputs[:, 1:2])
        
        return psnr_1, ssim_1, mae_1, psnr_2, ssim_2, mae_2

    
    def validation_end(self, val_outputs):
        
        psnr_1, ssim_1, mae_1, psnr_2, ssim_2, mae_2 = val_outputs
        psnr_1 = psnr_1.mean()
        ssim_1 = ssim_1.mean()
        psnr_2 = psnr_2.mean()
        ssim_2 = ssim_2.mean()
        mae_1 = mae_1.mean()
        mae_2 = mae_2.mean()
        print(f"psnr_1 is {psnr_1}, ssim_1 is {ssim_1}, psnr_2 is {psnr_2}, ssim_2 is {ssim_2}, mae_1 is {mae_1}, mae_2 is {mae_2}")
        
    
        self.log("psnr_1", psnr_1, step=self.epoch)
        self.log("ssim_1", ssim_1, step=self.epoch)
        self.log("psnr_2", psnr_2, step=self.epoch)
        self.log("ssim_2", ssim_2, step=self.epoch)
        self.log("mae_1", mae_1, step=self.epoch)
        self.log("mae_2", mae_2, step=self.epoch)

        self.log("mean_psnr", (psnr_1+psnr_2) / 2, step=self.epoch)
        self.log("mean_ssim", (ssim_1+ssim_2) / 2, step=self.epoch)
        self.log("mean_mae", (mae_1 + mae_2) / 2, step=self.epoch)

        m = (psnr_1 + psnr_2) / 2
        if m > self.best_mean_dice:
            self.best_mean_dice = m
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{m:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{m:.4f}.pt"), 
                                        delete_symbol="final_model")
        
if __name__ == "__main__":

    trainer = Pix2PixTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17752,
                            training_script=__file__)
    
    train_ds, val_ds, _ = get_train_val_dataset(image_size=image_size)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)