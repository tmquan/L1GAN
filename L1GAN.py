"""
To run this template just do:
python generative_adversarial_net.py
After a few epochs, launch TensorBoard to see the images being generated at every batch:
tensorboard --logdir default
"""
import os
import copy
from argparse import ArgumentParser
from collections import OrderedDict

import random
import pandas as pd
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.nn.functional import avg_pool2d
        
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from tensorpack.dataflow import imgaug
from tensorpack.dataflow import AugmentImageComponent
from tensorpack.dataflow import BatchData, MultiProcessRunner, PrintData, MapData, FixedSizeData
import albumentations as AB

from Data import MultiLabelCXRDataset
from Losses import *
def DiceScore(output, target, smooth=1.0, epsilon=1e-7, axis=(2, 3)):
    """
    Compute mean dice coefficient over all abnormality classes.
    Args:
        output (Numpy tensor): tensor of ground truth values for all classes.
                                    shape: (batch, num_classes, x_dim, y_dim)
        target (Numpy tensor): tensor of predictions for all classes.
                                    shape: (batch, num_classes, x_dim, y_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """
    y_true = target
    y_pred = output
    dice_numerator = 2*np.sum(y_true*y_pred, axis=axis) + epsilon
    dice_denominator = (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + epsilon)
    dice_coefficient = np.mean(dice_numerator / dice_denominator)

    return dice_coefficient

class SoftDiceLoss(nn.Module):
    def init(self):
        super(SoftDiceLoss, self).init()

    def forward(self, output, target, smooth=1.0, epsilon=1e-7, axis=(1)):
        """
        Compute mean soft dice loss over all abnormality classes.
        Args:
            y_true (Torch tensor): tensor of ground truth values for all classes.
                                        shape: (batch, num_classes, x_dim, y_dim)
            y_pred (Torch tensor): tensor of soft predictions for all classes.
                                        shape: (batch, num_classes, x_dim, y_dim)
            axis (tuple): spatial axes to sum over when computing numerator and
                          denominator in formula for dice loss.
                          Hint: pass this as the 'axis' argument to the K.sum
                                and K.mean functions.
            epsilon (float): small constant added to numerator and denominator to
                            avoid divide by 0 errors.
        Returns:
            dice_loss (float): computed value of dice loss.  
        """
        y_true = target
        y_pred = output
        dice_numerator = 2*torch.sum(y_true*y_pred, dim=axis) + epsilon
        dice_denominator = (torch.sum(y_true*y_true, dim=axis) + torch.sum(y_pred*y_pred, dim=axis) + epsilon)
        dice_coefficient = torch.mean(dice_numerator / dice_denominator)
        
        dice_loss = 1 - dice_coefficient
        return dice_loss

class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.init_size = self.hparams.shape // 16  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(self.hparams.latent_dim, 1024 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(1024), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),

            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



class Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.discrim = getattr(torchvision.models, 'densenet121')(
            pretrained=True)
        self.discrim.features.conv0 = nn.Conv2d(3, 64, 
            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.discrim.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
        )
        print(self.discrim)

        self.adv_layer = nn.Sequential(nn.Linear(1024, 1), nn.Tanh())
        # self.aux_layer = nn.Sequential(nn.Linear(1024, self.hparams.types), nn.Tanh())

    def forward(self, img):
        out = self.discrim(img)
        pred = self.adv_layer(out) / 2.0 + 0.5
        # prob = self.aux_layer(out) / 2.0 + 0.5

        return pred

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class L1GAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = torch.device("cuda")
    
        self.gen = Generator(hparams).to(self.device)
        self.dis = Discriminator(hparams).to(self.device)
        self.gen.apply(weights_init_normal)
        self.dis.apply(weights_init_normal)

        self.adv_loss = torch.nn.BCELoss() #SoftDiceLoss() #torch.nn.BCELoss()
        # self.probability_loss = torch.nn.BCELoss() #SoftDiceLoss() #torch.nn.BCELoss()

        # cache for generated images
        self.fake_imgs = None
        self.real_imgs = None

        self.loss_fn = LSGAN(dis = self.dis)
        
    def forward(self, z):
        return self.gen(z)

    # def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
    #                    second_order_closure=None):
    #     # print(optimizer.step())
    #     # print(1e4*optimizer.param_groups[0]['lr'])
    #     if optimizer_idx==0:
    #         self.lr_scale = 1e8
    #         print(optimizer.param_groups[0]['lr'])
    #         self.lr_scale *= optimizer.param_groups[0]['lr']
            
    #     if optimizer_idx==1:
    #         print(optimizer.param_groups[0]['lr'])
    #         self.lr_scale *= optimizer.param_groups[0]['lr']
    #         print(self.lr_scale)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, lbls = batch
        imgs = imgs.to(self.device) / 128.0 - 1.0
        lbls = lbls.to(self.device)
        self.real_imgs = imgs

        batchs = imgs.shape[0]
        # sample some random latent points
        z = torch.randn(batchs, self.hparams.latent_dim).to(self.device)
        fake_imgs = self.gen(z)

        # Calculate w1 and w2
        e1 = self.hparams.e1 
        e2 = self.hparams.e2
        assert e2 > e1
        ep = self.current_epoch 
        if ep < e1:
            w1 = 1
            w2 = 0
        elif ep > e2:
            w1 = 0
            w2 = 1
        else:
            w2 = (ep-e1) / (e2-e1)
            w1 = (e2-ep) / (e2-e1)

        # w2 = 1
        # train generator
        if optimizer_idx == 0:
            # fake = torch.ones((batchs, 1)).to(self.device)
            # fake_pred = self.dis(fake_imgs)
            # fake_loss = self.adv_loss(fake_pred, fake)  
            # fake_loss = -torch.mean(fake_pred)
            # ell1_loss = nn.L1Loss()(fake_imgs, imgs)
            ell1_loss = torch.mean(torch.abs(fake_imgs - imgs) / 2.0)
            g_loss = w2*self.loss_fn.gen_loss(imgs, fake_imgs) + w1*ell1_loss

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train dis
        if optimizer_idx == 1:
            # # how well can it label as real?
            # true = torch.ones((batchs, 1)).to(self.device)
            # real_pred = self.dis(imgs)
            # real_loss = self.adv_loss(real_pred, true) 
            # # real_loss = -torch.mean(real_pred)

            # # how well can it label as fake?
            # fake = torch.zeros((batchs, 1)).to(self.device)
            # fake_pred = self.dis(fake_imgs.detach())
            # fake_loss = self.adv_loss(fake_pred, fake)
            # # fake_loss = torch.mean(fake_pred)

            # # dis loss is the average of these
            # d_loss = (real_loss + fake_loss) / 2
            d_loss = self.loss_fn.dis_loss(imgs, fake_imgs)

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(b1, b2))

        # gen_sched = {'scheduler': ExponentialLR(gen_opt, 0.99),
        #                          'interval': 'step'}  # called after each training step
        # dis_sched = CosineAnnealing(discriminator_opt, T_max=10) # called every epoch
        # sch_g = {'scheduler': torch.optim.lr_scheduler.StepLR(opt_g, step_size=11054, gamma=0.99999), #ExponentialLR(opt_g, 0.999999),
        #          'interval': 'step'}
        sch_g = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10),
                 'interval': 'step'}    
        sch_d = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10),
                 'interval': 'step'}  
        return [opt_g, opt_d], [sch_g, sch_d]

    def train_dataloader(self):
        ds_train = MultiLabelCXRDataset(folder=self.hparams.data,
                                        is_train='train',
                                        fname='covid_train_v5.csv',
                                        types=self.hparams.types,
                                        pathology=None,
                                        resize=int(self.hparams.shape),
                                        balancing=None)

        ds_train.reset_state()
        ag_train = [
            # imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.RandomChooseAug([
                imgaug.Albumentations(AB.Blur(blur_limit=4, p=0.25)),
                imgaug.Albumentations(AB.MotionBlur(blur_limit=4, p=0.25)),
                imgaug.Albumentations(AB.MedianBlur(blur_limit=4, p=0.25)),
            ]),
            imgaug.Albumentations(AB.CLAHE(tile_grid_size=(32, 32), p=0.5)),
            imgaug.RandomOrderAug([
                imgaug.Affine(shear=10, border=cv2.BORDER_CONSTANT, 
                    interp=cv2.INTER_AREA),
                imgaug.Affine(translate_frac=(0.01, 0.02), border=cv2.BORDER_CONSTANT, 
                    interp=cv2.INTER_AREA),
                imgaug.Affine(scale=(0.5, 1.0), border=cv2.BORDER_CONSTANT, 
                    interp=cv2.INTER_AREA),
            ]),
            imgaug.RotationAndCropValid(max_deg=10, interp=cv2.INTER_AREA),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.8, 1.0),
                                                aspect_ratio_range=(0.8, 1.2),
                                                interp=cv2.INTER_AREA, target_shape=self.hparams.shape),
            imgaug.ToFloat32(),
        ]
        ds_train = AugmentImageComponent(ds_train, ag_train, 0)

        ds_train = BatchData(ds_train, self.hparams.batch, remainder=True)
        if self.hparams.debug:
            ds_train = FixedSizeData(ds_train, 2)
        ds_train = MultiProcessRunner(ds_train, num_proc=4, num_prefetch=16)
        ds_train = PrintData(ds_train)
        ds_train = MapData(ds_train,
                           lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2))),
                                       torch.tensor(dp[1]).float()])
        return ds_train

    def on_epoch_end(self):
        batchs = 16
        z = torch.randn(batchs, self.hparams.latent_dim).to(self.device)
        # p = torch.empty(batchs, self.hparams.types).random_(2).to(self.device)

        # z = torch.cat([n, p*2-1], dim=1)
        # log sampled images
        self.fake_imgs = self.gen(z)

        grid = torchvision.utils.make_grid(self.fake_imgs[:batchs] / 2.0 + 0.5, normalize=True)
        self.logger.experiment.add_image(f'fake_imgs', grid, self.current_epoch)

        grid = torchvision.utils.make_grid(self.real_imgs[:batchs] / 2.0 + 0.5, normalize=True)
        self.logger.experiment.add_image(f'real_imgs', grid, self.current_epoch)

        self.viz = nn.Sequential(
            self.gen,
            self.dis
            )
        self.logger.experiment.add_graph(self.viz, z)



def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    if hparams.seed is not None:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hparams.load:
        model = L1GAN(hparams).load_from_checkpoint(hparams.load)
    else:
        model = L1GAN(hparams)

    custom_log_dir = os.path.join(str(hparams.save),
                                  str(hparams.note),
                                  str(hparams.shape),
                                  str(hparams.types),
                                  ),


    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # trainer = Trainer()
   
    trainer = Trainer(
        num_sanity_val_steps=0,
        default_root_dir=os.path.join(str(hparams.save),
                                  str(hparams.note),
                                  str(hparams.shape),
                                  str(hparams.types),
                                  ),
        default_save_path=os.path.join(str(hparams.save),
                                  str(hparams.note),
                                  str(hparams.shape),
                                  str(hparams.types),
                                  ),
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        # checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=1,
        early_stop_callback=None,
        # train_percent_check=hparams.percent_check,
        # val_percent_check=hparams.percent_check,
        # test_percent_check=hparams.percent_check,
        # distributed_backend=hparams.distributed_backend,
        # use_amp=hparams.use_16bit,
        # val_check_interval=hparams.val_check_interval,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    # trainer.fit(model)
    if hparams.eval:
        assert hparams.loadD
        model.eval()
        # trainer.test(model)
        pass
    elif hparams.pred:
        assert hparams.load
        model.eval()
        pass
    else:
        trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    # Training params
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.0, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")

    parser.add_argument('--data', metavar='DIR', default=".", type=str, help='path to dataset')
    parser.add_argument('--save', metavar='DIR', default="train_log", type=str, help='path to save output')
    parser.add_argument('--info', metavar='DIR', default="train_log", help='path to logging output')
    parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
    parser.add_argument('--seed', type=int, default=1, help='reproducibility')
    parser.add_argument('--note', default="custom", type=str, help='custom string')
    # Inference params
    parser.add_argument('--load', action='store_true', help='path to logging output')
    parser.add_argument('--pred', action='store_true', help='run predict')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')

    # Dataset params
    parser.add_argument("--e1", type=int, default=1, help="Epoch 1")
    parser.add_argument("--e2", type=int, default=20, help="Epoch 2")
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--types', type=int, default=-1)
    parser.add_argument('--shape', type=int, default=256)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--debug', action='store_true', help='use fast mode')
    
    hparams = parser.parse_args()

    main(hparams)