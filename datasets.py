from cmath import isnan
import enum
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch 

import json 
import os
import numpy as np
import cv2

from generate_dataset import MAX_SEQ_LEN
from helpers import angle_between


class COCOPolygonDataset(Dataset):
    def __init__(self, key_pts, output_angle=False):
        self.key_pts = json.loads(open(key_pts).read())
        self.out_transform = transforms.GaussianBlur(9, sigma=1)
        self.idxs = list(self.key_pts.keys())
        self.output_angle = output_angle

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])

        self.buffer = 10 # 10 px

    def __len__(self):
        return len(self.idxs) # should be the same

    def __getitem__(self, idx):
        key_pts_list = np.array(self.key_pts[self.idxs[idx]])

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

                key_pts_list[:, :, 0] = (r_mul * key_pts_list[:, :, 0].astype(np.float32)).astype(np.int8) 
                key_pts_list[:, :, 1] = (r_mul * key_pts_list[:, :, 1].astype(np.float32)).astype(np.int8)

                # centering
                mean = np.mean(key_pts_list, 1)[0]
                key_pts_list[0] = key_pts_list[0] + (np.array([[111, 111]]) - mean)

        in_img = np.zeros((224, 224))
        output_img = torch.zeros((1, 224, 224))

        for i in range(key_pts_list.shape[1]):
            key_pt = key_pts_list[0, i]
            key_pts_list[0, i] = [np.clip(key_pt[0], self.buffer, 223 - self.buffer), np.clip(key_pt[1], self.buffer, 223 - self.buffer)]
            key_pt = key_pts_list[0, i]
            output_img[0, key_pt[1], key_pt[0]] = 1

        in_img = cv2.fillPoly(in_img, key_pts_list, color=(255,255,255))

        length_oh = F.one_hot(torch.tensor([length - 1]), MAX_SEQ_LEN) # starts at zero
        key_pts_list = np.array(key_pts_list).reshape(length, 2)



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