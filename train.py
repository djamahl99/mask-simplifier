from cmath import isnan
from glob import glob
import json
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import trange, tqdm
import natsort
import wandb
import logging

import cv2

import pytorch_warmup as warmup
from datasets import COCOPolygonDataset

from generate_dataset import MAX_SEQ_LEN
from model import PolygonPredictor, ResNetUNet
from utils import dice_loss, dice_coeff

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle=None, true_angle=None, show=False):
    try:
        for i in range(min(15, in_imgs.shape[0])):
            plt.figure(figsize=(15,10))
            ##################
            plt.subplot(3,2,1)
            plt.title("Mask In")
            plt.imshow(in_imgs[i][0].cpu().detach())

            ##################
            plt.subplot(3,2,2)
            p_l = torch.argmax(pred_len[i]) + 1
            p = pred_pos[i][0].cpu().detach()
            plt.title(f"predict {p_l} vertices. range: [{p.min():.1f}, {p.max():.1f}]")
            p = (p - p.min()) / (p.max() - p.min())
            plt.imshow(p)

            ##################
            plt.subplot(3,2,3)
            plt.imshow(true_pos[i][0].cpu().detach())
            t_l = torch.argmax(true_len[i]) + 1
            plt.title(f"GT, has {t_l} vertices")

            ##################
            plt.subplot(3,2,4)
            plt.title("topk w/ predicted k (number of vertices)")
            _, indices = torch.topk(p.flatten(0), p_l)
            indices = (np.array(np.unravel_index(indices.numpy(), p.shape)).T)
            indices = indices.reshape(-1, 2)
            topk_image = np.zeros((224,224), dtype=float)
            for index in indices:
                topk_image[index[0], index[1]] = 1
            plt.imshow(topk_image)
            
            if pred_angle is not None and true_angle is not None:
                # ANGLE
                #################
                plt.subplot(3,2,5)
                plt.title("Angle Prediction")
                p = pred_angle[i][0].cpu().detach()
                p = (p - p.min()) / (p.max() - p.min())
                plt.imshow(p)

                #################
                plt.subplot(3,2,6)
                plt.title("GT Angle")
                p = true_angle[i][0].cpu().detach()
                p = (p - p.min()) / (p.max() - p.min())
                plt.imshow(p)

            if show:
                plt.show()
            else:
                plt.savefig(f"out/epoch{epoch}_{i}.png")
            plt.close()
    except:
        pass # was throwing errors hours into training

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for image, true_len, mask_true in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        mask_true = torch.cat([mask_true, 1 - mask_true], dim=1).to(device)

        with torch.no_grad():
            # predict the mask
            length, mask_pred = net(image)

            # convert to one-hot format
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            mask_pred = torch.cat([mask_pred, 1 - mask_pred], dim=1).to(device)
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score

    return dice_score / num_val_batches
    
def main():
    batch_size = 32
    epochs = 20
    lr = 1e-4
    load_model = False

    ####################################################################################
    # load dataset #####################################################################
    dataset = COCOPolygonDataset('key_pts/key_pts_instances_train2017.json')
    # dataset = COCOPolygonDataset('key_pts/key_pts_instances_val2017.json', output_angle=False) 
    # dataset = COCOPolygonDataset('key_pts/key_pts_instances_val2017.json', original_masks='annotations/instances_train2017.json')
    val_dataset = COCOPolygonDataset('key_pts/key_pts_instances_val2017.json')

    train_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size = batch_size,
                                        num_workers = 4,
                                        shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size = batch_size,
                                        num_workers = 0,
                                        shuffle = True)

    amp = False
    device = torch.device('cuda')

    # model = PolygonPredictor().to(device)
    model = ResNetUNet(1).to(device)

    model_save_name = f"{model._get_name()}.pt"
    writer.add_text("Model Name", model._get_name())
    writer.add_scalar("Learning Rate", lr)
    writer.add_scalar("Epochs", epochs)

    if os.path.exists(model_save_name) and load_model:
        model = torch.load(model_save_name).to(device)

    model.train(True)

    ####################################################################################
    # optimizers and lr scheduling ######################################################

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion_len = nn.BCELoss().to(device)
    criterion_pos = nn.BCELoss().to(device)
    criterion_angle = nn.MSELoss().to(device)

    global_step = 0

    for epoch in trange(epochs):     
        n = 0
        running_loss = 0
        running_pos = 0
        running_len = 0
        running_angle = 0
        
        with tqdm(desc=f"EPOCH {epoch}", unit='img') as pbar:
            for in_imgs, true_len, true_pos in tqdm(train_loader, unit='batch'):
                in_imgs = in_imgs.to(device=device, dtype=torch.float32)
                mask_true = torch.cat([true_pos, 1 - true_pos], dim=1).to(device)
                true_len = true_len.to(device, dtype=torch.float32).squeeze(1)
                # true_angle = true_angle.to(device, dtype=torch.float32)

                pred_len, pred_pos = model(in_imgs)
                pred_pos_2ch = torch.cat([pred_pos, 1 - pred_pos], dim=1).to(device)

                if (n == 10) and epoch == 0:
                    save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle=None, true_angle=None, show=True)

                loss_pos = criterion_pos(pred_pos_2ch, mask_true) + dice_loss(pred_pos_2ch, mask_true)
                loss_len = criterion_len(pred_len, true_len)
                # loss_angle = criterion_angle(pred_angle, true_angle)
                loss = loss_pos + loss_len #+ loss_angle

                
                running_loss += loss.item()
                running_pos += loss_pos.item()
                running_len += loss_len.item()
                running_angle += 0 #loss_angle.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n += 1
                global_step += 1

                if global_step % 1000 == 0:
                    # save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle=None, true_angle=None)
                    torch.save(model, model_save_name)

                    val_score = evaluate(model, val_loader, device)
                    writer.add_scalar("Dice Score/Val", val_score, global_step)
                    writer.add_scalar("Train/Cross Entropy", running_loss/n, global_step)

                    scheduler.step(val_score)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar("Train/BCE - Epoch", running_loss/n, global_step)
                    writer.add_scalar("Train/BCE + Dice: Position", running_pos/n, global_step)
                    writer.add_scalar("Train/BCE: num vertices", running_len/n, global_step)
                    writer.add_scalar("Train/MSE: Angle", running_angle/n, global_step)

                    writer.flush()

                pbar.update(in_imgs.shape[0])

                pbar.set_postfix(**{'loss(ave)': running_loss/n, 'loss(len)': running_len/n, 'loss(pos)': running_pos/n, 'loss(angle)': running_angle/n})

        # save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle=None, true_angle=None)
        torch.save(model, model_save_name)
        writer.flush()

    writer.close()

if __name__ == "__main__":
    main()