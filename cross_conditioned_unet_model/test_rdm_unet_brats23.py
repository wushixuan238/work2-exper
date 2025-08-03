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
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
data_dir = "./data/fullres/train"
fold = 0

logdir = f"./logs_brats23/diffusion_240_t1n_to_t1c_noflip"
env = "pytorch"
model_save_path = os.path.join(logdir, "model")
max_epoch = 50
batch_size = 16
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
        sd = self.filte_sd("/home/xingzhaohu/image_synthesis/logs_brats23/rdm_240_t1n_to_t1c_noflip/model/model.pt")
        rdm_model.load_state_dict(sd)
        # rdm_model.load_state_dict(torch.load(, map_location="cpu"))
        rdm_model.to(self.device)

        self.sampler = DDIMSampler(model=rdm_model)

        self.rdm_model = rdm_model

        self.model = UNetModel(image_size=240, in_channels=2,
                  model_channels=96, out_channels=2, 
                  num_res_blocks=1, attention_resolutions=[32,16,8],
                  channel_mult=[1, 1, 2, 2])

        # self.load_state_dict("/home/xingzhaohu/image_synthesis/logs_brats23/rdm_unet_240/model/final_model_37.9970.pt")
        # self.load_state_dict("/home/xingzhaohu/image_synthesis/logs_brats23/rdm_unet_240/model/final_model_36.0747.pt")
        self.load_state_dict("/home/xingzhaohu/image_synthesis/logs_brats23_norm01_filter_data/rdm_unet_240_ep20_sample30/model/final_model_31.9359.pt")
        self.best_mean_dice = 0.0

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=3e-5, eps=1e-8)
        # self.scheduler_type = "poly"
        self.loss_func = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.index = 0

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
        
        pred = self.model(batch["t1n"])

        loss = self.mse(pred, batch["t1c"])
        del batch

        self.log("loss", loss, step=self.global_step)
        return loss
    

    def cal_metric(self, pred, gt):
        pred = pred.clamp(0, 1)

        pred = pred.cpu().numpy()[0]
        gt = gt.cpu().numpy()[0]
        # pred = (pred + 1) / 2 * 255
        # gt = (gt + 1) / 2 * 255

        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)
        mae = np.mean(np.abs(pred - gt))
        return psnr, ssim, mae  

    
    def save_image(self, pred, save_path):
        pred = Image.fromarray(pred).convert("L")

        pred.save(save_path)

    def validation_step(self, batch):
        seg = batch["seg"]
        image_t1n, image_t1c, image_t2w, image_t2f = batch["t1n"], batch["t1c"], batch["t2w"], batch["t2f"]
        inputs = torch.cat([image_t1n, image_t2w], dim=1)

        outputs = torch.cat([image_t1c, image_t2f], dim=1)
        
        sampled_rep, _ = self.sampler.sample(40, conditioning=None, batch_size=1,
                                                                        shape=(192, 1, 1),
                                                                        eta=1.0, verbose=False)
        sampled_rep = sampled_rep.squeeze(-1).squeeze(-1)

        pred = self.model(inputs, sampled_rep)

        self.index += 1

        # psnr_1, ssim_1, mae_1 = self.cal_metric(pred[:, 0:1], outputs[:, 0:1])
        # psnr_2, ssim_2, mae_2 = self.cal_metric(pred[:, 1:2], outputs[:, 1:2])
        
        # print(psnr_1, ssim_1, mae_1, "|", psnr_2, ssim_2, mae_2)
        pred_t1c = pred[:, 0:1]
        # pred_t1c = pred_t1c * 0.5 + 0.5
        pred_t1c = pred_t1c.clamp(0, 1) * 255
        pred_t1c = pred_t1c.cpu().numpy()[0, 0]

        pred_t2f = pred[:, 1:2]
        # pred_t2f = pred_t2f * 0.5 + 0.5
        pred_t2f = pred_t2f.clamp(0, 1) * 255
        pred_t2f = pred_t2f.cpu().numpy()[0, 0]
        
        
        index = batch["index"][0]
        identifier = batch["identifier"][0]
        save_dir = f"./save_t1c_t2f_for_seg_filter_data/rdm_unet_norm01_ep20/{identifier}/"
        os.makedirs(save_dir, exist_ok=True)

        print(f"index is {index}, identifier is {identifier}")

        self.save_image(pred_t1c, save_path=os.path.join(save_dir, f"{index}_t1c.png"))
        self.save_image(pred_t2f, save_path=os.path.join(save_dir, f"{index}_t2f.png"))

        # image_t1c = image_t1c * 0.5 + 0.5
        image_t1c = image_t1c.clamp(0, 1) * 255

        # image_t2f = image_t2f * 0.5 + 0.5
        image_t2f = image_t2f.clamp(0, 1) * 255
        image_t1c = image_t1c.cpu().numpy()[0, 0]
        image_t2f = image_t2f.cpu().numpy()[0, 0]

        self.save_image(image_t1c, save_path=os.path.join(save_dir, f"{index}_t1c_gt.png"))
        self.save_image(image_t2f, save_path=os.path.join(save_dir, f"{index}_t2f_gt.png"))

        # gt = batch["t1c"]
        # gt = gt.cpu().numpy()[0, 0]

        # image_t1n = image_t1n.cpu().numpy()[0, 0]
        # image_t2w = image_t2w.cpu().numpy()[0, 0]
        # image_t1c = image_t1c.cpu().numpy()[0, 0]
        # image_t2f = image_t2f.cpu().numpy()[0, 0]


        # if index > 60 and index < 110:
        # if seg.sum() > 50:
        #     import matplotlib.pyplot as plt
        #     plt.subplot(2, 3, 1)
        #     plt.imshow(image_t1n, cmap="gray")
        #     plt.subplot(2, 3, 2)
        #     plt.imshow(pred_t1c, cmap="gray")
        #     plt.subplot(2, 3, 3)
        #     plt.imshow(image_t1c, cmap="gray")

        #     plt.subplot(2, 3, 4)
        #     plt.imshow(image_t2w, cmap="gray")
        #     plt.subplot(2, 3, 5)
        #     plt.imshow(pred_t2f, cmap="gray")
        #     plt.subplot(2, 3, 6)
        #     plt.imshow(image_t2f, cmap="gray")

        #     save_dir = "./rdm_unet_brats23_pred"
        #     os.makedirs(save_dir, exist_ok=True)
        #     plt.savefig(os.path.join(save_dir, f"{self.index}.png"))

        # import time 
        # time.sleep(1)
        # return psnr_1, ssim_1, mae_1, psnr_2, ssim_2, mae_2

        return 1
        
    
    def validation_end(self, val_outputs):
        psnr, ssim = val_outputs
        psnr = psnr.mean()
        ssim = ssim.mean()
        print(f"psnr is {psnr}, ssim is {ssim}")
        
    
        self.log("psnr", psnr, step=self.epoch)
        self.log("ssim", ssim, step=self.epoch)

        if psnr > self.best_mean_dice:
            self.best_mean_dice = psnr
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{psnr:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{psnr:.4f}.pt"), 
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
    
    train_ds, val_ds, seg_ds = get_train_val_dataset(image_size=image_size)

    # trainer.train(train_dataset=train_ds, val_dataset=val_ds)
    trainer.validation_single_gpu(seg_ds)


    # import matplotlib.pyplot as plt 
    # index = 0

    # def process(image, mask):
    #     image = image.permute(1, 2, 0)
    #     # mask = mask[0]
    #     return image, mask 
    
    # for d in train_ds:
    #     index += 1

    #     if index < 120:
    #         continue
    #     image = d["image"]
    #     mask = d["mask"]

    #     print(image.shape)
    #     print(mask.shape)
    #     image_1 = image[0]
    #     image_2 = image[1]
    #     image_3 = image[2]
    #     mask_1 = mask[0]
    #     mask_2 = mask[0]
    #     mask_3 = mask[0]

    #     # print()
    #     # print(f"mask shape is {mask.shape}")
    #     # mask = mask[0]
    #     # image = image.permute(1, 2, 0)
    #     image_1, mask_1 = process(image_1, mask_1)
    #     image_2, mask_2 = process(image_2, mask_2)
    #     image_3, mask_3 = process(image_3, mask_3)
    #     plt.subplot(3, 2, 1)
    #     plt.imshow(image_1)
    #     plt.subplot(3, 2, 2)
    #     plt.imshow(mask_1)
    #     plt.subplot(3, 2, 3)
    #     plt.imshow(image_2)
    #     plt.subplot(3, 2, 4)
    #     plt.imshow(mask_2)
    #     plt.subplot(3, 2, 5)
    #     plt.imshow(image_3)
    #     plt.subplot(3, 2, 6)
    #     plt.imshow(mask_3)
    #     plt.show()
    #     if index == 130:
    #         break