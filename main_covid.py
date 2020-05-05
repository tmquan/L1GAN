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

from pprint import pprint
import sklearn.metrics
from sklearn.utils import resample
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
import torchxrayvision as xrv

from Data import MultiLabelCXRDataset
from Losses import *

from fastai import *
from fastai.vision import *
from fastai.callbacks import *

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
    dice_denominator = (np.sum(y_true, axis=axis) +
                        np.sum(y_pred, axis=axis) + epsilon)
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
        dice_denominator = (torch.sum(y_true*y_true, dim=axis) +
                            torch.sum(y_pred*y_pred, dim=axis) + epsilon)
        dice_coefficient = torch.mean(dice_numerator / dice_denominator)

        dice_loss = 1 - dice_coefficient
        return dice_loss


import torch.utils.model_zoo as model_zoo

# Optional list of dependencies required by the package
dependencies = ['torch']


def PGAN(pretrained=False, *args, **kwargs):
    """
    Progressive growing model
    pretrained (bool): load a pretrained model ?
    model_name (string): if pretrained, load one of the following models
    celebaHQ-256, celebaHQ-512, DTD, celeba, cifar10. Default is celebaHQ.
    """
    from models.progressive_gan import ProgressiveGAN as PGAN
    if 'config' not in kwargs or kwargs['config'] is None:
        kwargs['config'] = {}

    model = PGAN(useGPU=kwargs.get('useGPU', True),
                 storeAVG=True,
                 **kwargs['config'])

    checkpoint = {"celebAHQ-256": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ_s6_i80000-6196db68.pth',
                  "celebAHQ-512": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ16_december_s7_i96000-9c72988c.pth',
                  "DTD": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/testDTD_s5_i96000-04efa39f.pth',
                  "celeba": "https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s5_i83000-2b0acc76.pth"}
    if pretrained:
        if "model_name" in kwargs:
            if kwargs["model_name"] not in checkpoint.keys():
                raise ValueError("model_name should be in "
                                    + str(checkpoint.keys()))
        else:
            print("Loading default model : celebaHQ-256")
            kwargs["model_name"] = "celebAHQ-256"
        state_dict = model_zoo.load_url(checkpoint[kwargs["model_name"]],
                                        map_location='cpu')
        model.load_state_dict(state_dict)
    return model


def DCGAN(pretrained=False, *args, **kwargs):
    """
    DCGAN basic model
    pretrained (bool): load a pretrained model ? In this case load a model
    trained on fashionGen cloth
    """
    from models.DCGAN import DCGAN
    if 'config' not in kwargs or kwargs['config'] is None:
        kwargs['config'] = {}

    model = DCGAN(useGPU=kwargs.get('useGPU', True),
                  storeAVG=True,
                  **kwargs['config'])

    checkpoint = 'https://dl.fbaipublicfiles.com/gan_zoo/DCGAN_fashionGen-1d67302.pth'
    if pretrained:
        state_dict = model_zoo.load_url(checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
    return model

use_gpu = True if torch.cuda.is_available() else False

model = PGAN(pretrained=True, model_name="celebAHQ-512", useGPU=use_gpu)

class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.gen = model.getNetG()
        print(self.gen)
        self.gen.formatLayer.module = nn.Linear(in_features=hparams.latent_dim+hparams.types, 
            out_features=8192, bias=True)

    def forward(self, noise):
        return self.gen(noise)

class Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.dis = model.getNetD()
        print(self.dis)
        self.dis.decisionLayer = nn.Identity(inplace=True) #Linear(in_features=512, out_features=1+hparams.t, bias=True)
        
        self.adv_layer = nn.Sequential(nn.Linear(512, 1), nn.Tanh())
        self.aux_layer = nn.Sequential(nn.Linear(512, hparams.types), nn.Tanh())

    def forward(self, img):
        out = self.dis(img)
        pred = self.adv_layer(out) / 2.0 + 0.5
        return pred
        
    def classify(self, img):
        out = self.dis(img)
        prob = self.aux_layer(out) / 2.0 + 0.5
        return prob
        

class L1GAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = torch.device("cuda")

        self.gen = Generator(hparams).to(self.device)
        self.dis = Discriminator(hparams).to(self.device)
        # self.gen.apply(weights_init_normal)
        # self.dis.apply(weights_init_normal)

        # self.adv_loss = SoftDiceLoss() #torch.nn.BCELoss()
        # self.bce_loss = SoftDiceLoss() #torch.nn.BCELoss()


        # cache for generated images
        self.fake_imgs = None
        self.real_imgs = None

        self.loss_fn = LSGAN(dis=self.dis)
        # self.loss_fn = WGAN_GP(dis=self.dis)
    def forward(self, z):
        return self.gen(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, lbls = batch
        # imgs = imgs.to(self.device) / 128.0 - 1.0
        lbls = lbls.to(self.device)
        self.real_imgs = imgs

        batchs = imgs.shape[0]
        # sample some random latent points
        n = torch.randn(batchs, self.hparams.latent_dim).to(self.device)
        p = torch.empty(batchs, self.hparams.types).random_(2).to(self.device)
        z = torch.cat([n, p*2-1], dim=1)

        fake_imgs = self.gen(z)

        # train generator
        if optimizer_idx == 0:
            
            pred = self.dis.classify(fake_imgs)

            g_loss = self.loss_fn.gen_loss(imgs, fake_imgs)

            if self.hparams.note=='warmup':
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
                ell1_loss = torch.nn.L1Loss()(fake_imgs, imgs) #torch.mean(torch.abs(fake_imgs - imgs))
                g_loss *= w2
                g_loss += w1*ell1_loss 

            if self.hparams.case=='gendis':
                g_loss += torch.nn.BCELoss()(pred, p) + SoftDiceLoss()(pred, p)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train dis
        if optimizer_idx == 1:
            # print(imgs.shape, fake_imgs.shape)
            images = torch.cat([imgs, fake_imgs], dim=0)
            labels = torch.cat([lbls, p], dim=0)

            output = self.dis.classify(images)
            
            d_loss = self.loss_fn.dis_loss(imgs, fake_imgs) \
                   + torch.nn.BCELoss()(output, labels) + SoftDiceLoss()(output, labels)
                   # + self.bce_loss(output, labels)

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

        sch_g = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10),
                 'interval': 'step'}
        sch_d = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10),
                 'interval': 'step'}
        return [opt_g, opt_d], [sch_g, sch_d]

    def train_val_dataloader(self):
        extra = [*zoom_crop(scale=(0.5, 1.3), p=0.5), 
                 *rand_resize_crop(self.hparams.shape, max_scale=1.3),
                 squish(scale=(0.9, 1.2), p=0.5),
                 tilt(direction=(0, 3), magnitude=(-0.3, 0.3), p=0.5),
                 # cutout(n_holes=(1, 5), length=(10, 30), p=0.1)
                 ]
        transforms = get_transforms(max_rotate=11, max_zoom=1.3, max_lighting=0.1, do_flip=False, 
                                    max_warp=0.15, p_affine=0.5, p_lighting=0.3, xtra_tfms=extra)
        transforms = list(transforms); 
        transforms[1] = []

        df=pd.read_csv(os.path.join(self.hparams.data, 'covid_train_v5.csv'))
        balancing='up'
        if balancing == 'up':
            df_majority = df[df[self.hparams.pathology]==0]
            df_minority = df[df[self.hparams.pathology]==1]
            print(df_majority[self.hparams.pathology].value_counts())
            df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=df_majority[self.hparams.pathology].value_counts()[0],    # to match majority class
                                     random_state=hparams.seed) # reproducible results

            df_upsampled = pd.concat([df_majority, df_minority_upsampled])
            df = df_upsampled

        dset = (
            ImageList.from_df(df=df, 
                path=os.path.join(self.hparams.data, 'data'), cols='Images')
            .split_by_rand_pct(0.0, seed=hparams.seed)
            .label_from_df(cols=['Covid', 'Airspace_Opacity', 'Consolidation', 'Pneumonia'], label_cls=MultiCategoryList)
            .transform(transforms, size=self.hparams.shape, padding_mode='zeros')
            .databunch(bs=self.hparams.batch, num_workers=32)
            .normalize(imagenet_stats)
        )
        return dset.train_dl.dl, dset.valid_dl.dl

    def eval_dataloader(self):
        extra = [*zoom_crop(scale=(0.5, 1.3), p=0.5), 
                 *rand_resize_crop(self.hparams.shape, max_scale=1.3),
                 squish(scale=(0.9, 1.2), p=0.5),
                 tilt(direction=(0, 3), magnitude=(-0.3, 0.3), p=0.5),
                 # cutout(n_holes=(1, 5), length=(10, 30), p=0.1)
                 ]
        transforms = get_transforms(max_rotate=11, max_zoom=1.3, max_lighting=0.1, do_flip=False, 
                                    max_warp=0.15, p_affine=0.5, p_lighting=0.3, xtra_tfms=extra)
        transforms = list(transforms); 
        transforms[1] = []

        df=pd.read_csv(os.path.join(self.hparams.data, 'covid_test_v5.csv'))

        dset = (
            ImageList.from_df(df=df, 
                path=os.path.join(self.hparams.data, 'data'), cols='Images')
            .split_by_rand_pct(1.0, seed=hparams.seed)
            .label_from_df(cols=['Covid', 'Airspace_Opacity', 'Consolidation', 'Pneumonia'], label_cls=MultiCategoryList)
            .transform(transforms, size=self.hparams.shape, padding_mode='zeros')
            .databunch(bs=self.hparams.batch, num_workers=32)
            .normalize(imagenet_stats)
        )
        return dset.train_dl.dl, dset.valid_dl.dl

    def train_dataloader(self):
        ds_train, ds_valid = self.train_val_dataloader()
        return ds_train
        
    def on_epoch_end(self):
        batchs = 10
        n = torch.randn(batchs, self.hparams.latent_dim).to(self.device)
        p = torch.empty(batchs, self.hparams.types).random_(2).to(self.device)
        z = torch.cat([n, p*2-1], dim=1)

        # log sampled images
        self.fake_imgs = self.gen(z)

        grid = torchvision.utils.make_grid(
            self.fake_imgs[:batchs] / 2.0 + 0.5, normalize=True, nrow=5)
        self.logger.experiment.add_image(f'fake_imgs', grid, self.current_epoch)

        grid = torchvision.utils.make_grid(
            self.real_imgs[:batchs] / 2.0 + 0.5, normalize=True, nrow=5)
        self.logger.experiment.add_image(f'real_imgs', grid, self.current_epoch)

        self.viz = nn.Sequential(
            self.gen,
            self.dis
        )
        self.logger.experiment.add_graph(self.viz, z)

    def val_dataloader(self):
        ds_train, ds_valid = self.eval_dataloader()
        return ds_valid
      

    def test_dataloader(self):
        pass

    def custom_step(self, batch, batch_idx, prefix='val'):
        image, target = batch
        output = self.dis.classify(image)
        loss = torch.nn.BCELoss()(output, target) + SoftDiceLoss()(output, target)

        result = OrderedDict({
            f'{prefix}_loss': loss,
            f'{prefix}_output': output,
            f'{prefix}_target': target,
        })
        # self.logger.experiment.add_image(f'{prefix}_images',
        #                                  torchvision.utils.make_grid(images / 255.0),
        #                                  dataformats='CHW')
        return result

    def validation_step(self, batch, batch_idx, prefix='val'):
        return self.custom_step(batch, batch_idx, prefix=prefix)

    def test_step(self, batch, batch_idx, prefix='test'):
        return self.custom_step(batch, batch_idx, prefix=prefix)

    def custom_epoch_end(self, outputs, prefix='val'):
        loss_mean = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()

        np_output = torch.cat([x[f'{prefix}_output'].squeeze_(0) for x in outputs], dim=0).to('cpu').numpy()
        np_target = torch.cat([x[f'{prefix}_target'].squeeze_(0) for x in outputs], dim=0).to('cpu').numpy()

        # print(np_output)
        # print(np_target)
        # Casting to binary
        np_output = 1 * (np_output >= self.hparams.threshold).astype(np.uint8)
        np_target = 1 * (np_target >= self.hparams.threshold).astype(np.uint8)

        print(np_target.shape)
        print(np_output.shape)

        result = {}
        result[f'{prefix}_loss'] = loss_mean

        tqdm_dict = {}
        tqdm_dict[f'{prefix}_loss'] = loss_mean

        tb_log = {}
        tb_log[f'{prefix}_loss'] = loss_mean

        f1_scores = []
        np_log = []
        if np_output.shape[0] > 0 and np_target.shape[0] > 0:
            for p in range(self.hparams.types):
                PP = np.sum((np_target[:,p] == 1))
                NN = np.sum((np_target[:,p] == 0))
                TP = np.sum((np_target[:,p] == 1) & (np_output[:,p] == 1))
                TN = np.sum((np_target[:,p] == 0) & (np_output[:,p] == 0))
                FP = np.sum((np_target[:,p] == 0) & (np_output[:,p] == 1))
                FN = np.sum((np_target[:,p] == 1) & (np_output[:,p] == 0))
                np_log.append([p, PP, NN, TP, TN, FP, FN])
                precision_score = (TP / (TP + FP + 1e-12))
                recall_score = (TP / (TP + FN + 1e-12))
                beta = 1
                f1_score = (1 + beta**2) * precision_score * recall_score / (beta**2 * precision_score + recall_score + 1e-12)
                beta = 2
                f2_score = (1 + beta**2) * precision_score * recall_score / (beta**2 * precision_score + recall_score + 1e-12)
                # f1_score = sklearn.metrics.fbeta_score(np_target[:, p], np_output[:, p], beta=1, average='macro')
                # f2_score = sklearn.metrics.fbeta_score(np_target[:, p], np_output[:, p], beta=2, average='macro')
                # precision_score = sklearn.metrics.precision_score(np_target[:, p], np_output[:, p], average='macro')
                # recall_score = sklearn.metrics.recall_score(np_target[:, p], np_output[:, p], average='macro')

                f1_scores.append(f1_score)
                tqdm_dict[f'{prefix}_f1_score_{p}'] = f'{f1_score:0.4f}'
                tqdm_dict[f'{prefix}_f2_score_{p}'] = f'{f2_score:0.4f}',
                tqdm_dict[f'{prefix}_precision_score_{p}'] = f'{precision_score:0.4f}'
                tqdm_dict[f'{prefix}_recall_score_{p}'] = f'{recall_score:0.4f}'

                tb_log[f'{prefix}_f1_score_{p}'] = f1_score
                tb_log[f'{prefix}_f2_score_{p}'] = f2_score
                tb_log[f'{prefix}_precision_score_{p}'] = precision_score
                tb_log[f'{prefix}_recall_score_{p}'] = recall_score

            tqdm_dict[f'{prefix}_f1_score_mean'] = f'{np.array(f1_scores).mean():0.4f}'
            tb_log[f'{prefix}_f1_score_mean'] = np.array(f1_scores).mean()
        pprint(np.array(np_log))
        pprint(tqdm_dict)
        result['log'] = tb_log
        np_output = []
        np_target = []
        return result

    def validation_epoch_end(self, outputs, prefix='val'):
        return self.custom_epoch_end(outputs, prefix=prefix)

    def test_epoch_end(self, outputs, prefix='test'):
        return self.custom_epoch_end(outputs, prefix=prefix)


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
                                  str(hparams.case),
                                  str(hparams.shape),
                                  str(hparams.types),
                                  ),

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # trainer = Trainer()
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(str(hparams.save),
                              str(hparams.note),
                              str(hparams.case),
                              str(hparams.pathology),
                              str(hparams.shape),
                              str(hparams.types),
                              # str(hparams.folds),
                              # str(hparams.valid_fold_index),
                              'ckpt'),
        save_top_k=hparams.epochs, #10,
        verbose=True,
        monitor='val_f1_score_mean' if hparams.pathology=='All' else 'val_f1_score_0',  # TODO
        mode='max'
    )

    trainer = Trainer(
        num_sanity_val_steps=0,
        default_root_dir=os.path.join(str(hparams.save),
                                      str(hparams.note),
                                      str(hparams.case),
                                      str(hparams.pathology),
                                      str(hparams.shape),
                                      str(hparams.types),
                                      ),
        default_save_path=os.path.join(str(hparams.save),
                                       str(hparams.note),
                                       str(hparams.case),
                                       str(hparams.pathology),
                                       str(hparams.shape),
                                       str(hparams.types),
                                       ),
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=1,
        early_stop_callback=None,
        fast_dev_run=hparams.fast_dev_run,
        # train_percent_check=hparams.percent_check,
        # val_percent_check=hparams.percent_check,
        # test_percent_check=hparams.percent_check,
        # distributed_backend=hparams.distributed_backend,
        # use_amp=hparams.use_16bit,
        val_check_interval=hparams.percent_check,
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--b1", type=float, default=0.0,)
    parser.add_argument("--b2", type=float, default=0.99)
    parser.add_argument("--latent_dim", type=int, default=64)

    parser.add_argument('--data', metavar='DIR', default=".", type=str)
    parser.add_argument('--save', metavar='DIR', default="train_log", type=str)
    parser.add_argument('--info', metavar='DIR', default="train_log")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--note', type=str, default="warmup")
    parser.add_argument('--case', type=str, default="gendis")
    # Inference params
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--pred', action='store_true')
    parser.add_argument('--eval', action='store_true')

    # Dataset params
    parser.add_argument("--fast_dev_run", action='store_true')
    parser.add_argument("--percent_check", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--e1", type=int, default=0)
    parser.add_argument("--e2", type=int, default=10)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--types', type=int, default=-1)
    parser.add_argument('--shape', type=int, default=256)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pathology', default='All')
    hparams = parser.parse_args()

    main(hparams)
