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

from guided_diffusion.unet_rdm import UNetModel
 

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
data_dir = "./data/fullres/train"
fold = 0

logdir = f"./logs_brats23/rdm_240_norm01_decouple"
env = "DDP"
model_save_path = os.path.join(logdir, "model")

max_epoch = 50
batch_size = 16
val_every = 1
num_gpus = 4
device = "cuda:0"
image_size = 224

class Pix2PixTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        from rcg.rdm.util import instantiate_from_config
        from omegaconf import OmegaConf

        config = OmegaConf.load("/home/xingzhaohu/image_synthesis/config/rdm/mocov3vitb_simplemlp_l12_w1536.yaml")

        self.model = instantiate_from_config(config.model)

        self.best_mean_dice = 1000000

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=3e-5, eps=1e-8)
        self.scheduler_type = "poly"
        self.loss_func = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
 
    def training_step(self, batch):
        _, image_t1c, _, image_t2f = batch["t1n"], batch["t1c"], batch["t2w"], batch["t2f"]
        # inputs = torch.cat([image_t1n, image_t2w], dim=1)
    
        outputs = torch.cat([image_t1c, image_t2f], dim=1)
    
        # img2 = batch["t1c"]
        batch = {"image": outputs, "class_label": torch.zeros((batch_size,)).long()}
        # self.model.set_input(batch)
        loss, _ = self.model(x=None, c=None, batch=batch)

        self.log("loss", loss, step=self.global_step)

        return loss 
    
    def cal_metric(self, pred, gt):
        pred = pred.clamp(0, 1)

        pred = pred.cpu().numpy()[0]
        gt = gt.cpu().numpy()[0]
        
        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)

        return psnr, ssim 
    
    def validation_step(self, batch):
        _, image_t1c, _, image_t2f = batch["t1n"], batch["t1c"], batch["t2w"], batch["t2f"]
        # inputs = torch.cat([image_t1n, image_t2w], dim=1)
    
        outputs = torch.cat([image_t1c, image_t2f], dim=1)
        batch = {"image": outputs, "class_label": torch.zeros((batch_size,)).long()}
        # self.model.set_input(batch)
        loss, _ = self.model(x=None, c=None, batch=batch)

        return loss.item()
    
    def validation_end(self, val_outputs):
        
        loss = val_outputs
        loss = loss.mean()
    
        print(f"loss is {loss}")
        
    
        self.log("loss_val_epoch", loss, step=self.epoch)

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"model.pt"), 
                                        )

        # torch.save(self.model.state_dict(), os.path.join(model_save_path, 
                                                        #  f"model.pt"))
        # if psnr > self.best_mean_dice:
        #     self.best_mean_dice = psnr
        #     save_new_model_and_delete_last(self.model, 
        #                                     os.path.join(model_save_path, 
        #                                     f"best_model_{psnr:.4f}.pt"), 
        #                                     delete_symbol="best_model")

        # save_new_model_and_delete_last(self.model, 
        #                                 os.path.join(model_save_path, 
        #                                 f"final_model_{psnr:.4f}.pt"), 
        #                                 delete_symbol="final_model")

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