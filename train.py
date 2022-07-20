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

from scipy.stats import norm

import cv2

import pytorch_warmup as warmup

from generate_dataset import MAX_SEQ_LEN
from model import PolygonPredictor
from utils import dice_loss, dice_coeff

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle=None, true_angle=None, show=False):
    for i in range(min(15, in_imgs.shape[0])):
        plt.figure(figsize=(15,10))
        ##################
        plt.subplot(3,2,1)
        plt.title("Mask In")
        plt.imshow(in_imgs[i][0].cpu().detach())

        ##################
        plt.subplot(3,2,2)
        p_l = torch.argmax(pred_len[i])
        p = pred_pos[i][0].cpu().detach()
        plt.title(f"predict {p_l} vertices. range: [{p.min():.1f}, {p.max():.1f}]")
        p = (p - p.min()) / (p.max() - p.min())
        plt.imshow(p)

        ##################
        plt.subplot(3,2,3)
        plt.imshow(true_pos[i][0].cpu().detach())
        p_l = torch.argmax(true_len[i])
        plt.title(f"GT, has {p_l} vertices")

        ##################
        plt.subplot(3,2,4)
        plt.title("topk w/ GT k")
        _, indices = torch.topk(p.flatten(0), p_l)
        indices = (np.array(np.unravel_index(indices.numpy(), p.shape)).T)
        indices = indices.reshape(-1, 2)
        topk_image = np.zeros((224,224), dtype=float)
        for index in indices:
            topk_image[index[0], index[1]] = 1
        plt.imshow(topk_image)
        
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

def euclid_dis(from_, to_):
    return np.sqrt(np.square(to_.astype(np.float32) - from_.astype(np.float32)).sum())

def normalized_polygon(vertices, tensor, sd=1):
    def draw_between(from_, to_, tensor):
        # print(f"from {from_} to {to_}")
        distance = euclid_dis(from_, to_)

        vec = to_ - from_
        for i in range(int(distance + 1)):
            current_pos = ((i / distance) * vec + from_).astype(int)
            closest_vert_dist = min(i, distance - i)

            l = norm.pdf(closest_vert_dist, scale=sd)

            tensor[0, np.clip(current_pos[0], 0, 223), np.clip(current_pos[1], 0, 223)] = l

    last_vertex = None
    vertices = vertices.reshape(-1, 2)
    first_vert = np.array([vertices[0][1], vertices[0][0]])
    for vertex in vertices:
        x, y = vertex[1], vertex[0]

        if last_vertex is None:
            last_vertex = np.array([x, y])
            continue 
        
        to_ = np.array([x, y])
        from_ = last_vertex

        draw_between(from_, to_, tensor)

        last_vertex = to_

    draw_between(last_vertex, first_vert, tensor)
    

    return tensor

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
            length, mask_pred, angle = net(image)

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

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class OrderedBatchDataset(Dataset):
    def __init__(self, input_dir, input_transform, key_pts, output_angle=False):
        self.input_dir = input_dir
        self.input_transform = input_transform
        input_imgs = os.listdir(input_dir)
        # self.input_imgs = natsort.natsorted(input_imgs)
        self.key_pts = json.loads(open(key_pts).read())
        self.out_transform = transforms.GaussianBlur(9, sigma=1)
        self.idxs = list(self.key_pts.keys())
        self.output_angle = output_angle

    def __len__(self):
        return len(self.idxs) # should be the same

    def __getitem__(self, idx):
        key_pts_list = np.array(self.key_pts[self.idxs[idx]])

        # print("shape, ",  key_pts_list.shape)
        length = key_pts_list.shape[-2]

        if True:
            # centering
            mean = np.mean(key_pts_list, 1)[0]
            key_pts_list[0] = key_pts_list[0] + (np.array([[111, 111]]) - mean)

            # rescaling
            rescale_range = 112
            min_x, max_x = np.min(key_pts_list[:, :, 0]), np.max(key_pts_list[:, :, 0])
            range_x = max_x - min_x
            min_y, max_y = np.min(key_pts_list[:, :, 1]), np.max(key_pts_list[:, :, 1])
            range_y = max_y - min_y

            if min(range_x, range_y) == 0:
                key_pts_list = np.array([[[100, 100], [100, 200], [200, 200], [200, 100], [100, 100]]])
                length = 5
            elif min(range_x, range_y) < 50:
                r_mul = 200 / (max(range_x, range_y) + 1) # plus one to avoid zero division

                key_pts_list[:, :, 0] = (r_mul * key_pts_list[:, :, 0].astype(np.float)).astype(np.int) 
                key_pts_list[:, :, 1] = (r_mul * key_pts_list[:, :, 1].astype(np.float)).astype(np.int)

                # centering
                mean = np.mean(key_pts_list, 1)[0]
                key_pts_list[0] = key_pts_list[0] + (np.array([[111, 111]]) - mean)

        in_img = np.zeros((224, 224))
        in_img = cv2.fillPoly(in_img, key_pts_list, color=(255,255,255))

        output_img = torch.zeros((1, 224, 224))

        length_oh = F.one_hot(torch.tensor([length - 1]), MAX_SEQ_LEN) # starts at zero
        key_pts_list = np.array(key_pts_list).reshape(length, 2)

        for key_pt in key_pts_list:
            output_img[0, np.clip(key_pt[1], 0, 223), np.clip(key_pt[0], 0, 223)] = 1

        # ANGLEEE
        if self.output_angle:
            angle_img = torch.zeros((1, 224, 224))

            for i in range(len(key_pts_list)):
                key_pt = key_pts_list[i]
                last_vert, current_vert, next_vert = key_pts_list[(i - 1) % length], key_pts_list[i], key_pts_list[(i + 1) % length]

                vec1 = current_vert - last_vert
                vec2 = next_vert - current_vert
                angle = 0

                try:
                    angle = angle_between(vec1, vec2)
                except:
                    angle = 0

                if isnan(angle):
                    angle = 0

                angle_img[0, np.clip(key_pt[1], 0, 223), np.clip(key_pt[0], 0, 223)] = angle

        output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())

        if self.output_angle:
            return self.input_transform(in_img), length_oh, output_img, angle_img
        else:
            return self.input_transform(in_img), length_oh, output_img

def main():
    # add random augmentation etc
    resize_t = transforms.Compose([
        # T.GaussianBlur(3),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float)
    ])

    batch_size = 16
    # load images
    dataset = OrderedBatchDataset('masks/instances_train2017', resize_t, 'key_pts/key_pts_instances_train2017.json', output_angle=True)
    # dataset = OrderedBatchDataset('masks/instances_val2017', resize_t, 'key_pts/key_pts_instances_val2017.json', output_angle=True)
    val_dataset = OrderedBatchDataset('masks/instances_val2017', resize_t, 'key_pts/key_pts_instances_val2017.json')

    train_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size = batch_size,
                                        num_workers = 4,
                                        shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size = batch_size,
                                        num_workers = 0,
                                        shuffle = True)

    epochs = 20
    lr = 1e-3
    amp = False
    device = torch.device('cuda')

    model = PolygonPredictor().to(device)

    model_save_name = f"{model._get_name()}.pt"
    writer.add_text("Model Name", model._get_name())
    writer.add_scalar("Learning Rate", lr)
    writer.add_scalar("Epochs", epochs)

    # model = torch.load(model_save_name).to(device)
    model.train(True)


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
            for in_imgs, true_len, true_pos, true_angle in tqdm(train_loader, unit='batch'):
                in_imgs = in_imgs.to(device=device, dtype=torch.float32)
                mask_true = torch.cat([true_pos, 1 - true_pos], dim=1).to(device)
                true_len = true_len.to(device, dtype=torch.float32).squeeze(1)
                true_angle = true_angle.to(device, dtype=torch.float32)

                pred_len, pred_pos, pred_angle = model(in_imgs)
                pred_pos_2ch = torch.cat([pred_pos, 1 - pred_pos], dim=1).to(device)

                if (n == 10) and epoch == 0:
                    save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle, true_angle, show=True)

                loss_pos = criterion_pos(pred_pos_2ch, mask_true) + dice_loss(pred_pos_2ch, mask_true)
                loss_len = criterion_len(pred_len, true_len)
                loss_angle = criterion_angle(pred_angle, true_angle)
                loss = loss_pos + loss_len + loss_angle

                
                running_loss += loss.item()
                running_pos += loss_pos.item()
                running_len += loss_len.item()
                running_angle += loss_angle.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n += 1
                global_step += 1

                if global_step % 1000 == 0:
                    save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle, true_angle)
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



        save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle, true_angle)
        # print(f"EPOCH {epoch} Loss {running_loss/n:.8f}")
        torch.save(model, model_save_name)
        writer.flush()

    writer.close()

if __name__ == "__main__":
    main()