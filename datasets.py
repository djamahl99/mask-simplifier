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
    def __init__(self, key_pts, output_angle=False, original_masks=None):
        self.out_transform = transforms.GaussianBlur(9, sigma=1)
        self.output_angle = output_angle 
        self.instances = None

        if original_masks is not None:
            print("using original masks")
            self.instances = json.loads(open(original_masks).read())
            img_id_to_img = {img['id']: img for img in self.instances['images']}

            print("img id to img", img_id_to_img)

            instances_tmp = {}
            new_shape = (224, 224)

            for annot in self.instances['annotations']:
                img = img_id_to_img[annot['image_id']]

                w, h = img['width'], img['height']
                n_w, n_h = (w / max(w,h)) * new_shape[0], (h / max(w,h)) * new_shape[1]
                r_w, r_h = n_w / w, n_h / h

                try:
                    pts = np.array([[annot['segmentation'][0][2*i] * r_w, annot['segmentation'][0][2*i+1] * r_h] for i in range(len(annot['segmentation'][0]) // 2)])
                except Exception as e:
                    continue

                instances_tmp[annot['id']] = np.int32([pts])
                # print("points shape", instances_tmp[annot['id']].shape)


            self.key_pts = instances_tmp
            self.idxs = list(self.instances.keys())

        else: # use simplified version
            self.key_pts = json.loads(open(key_pts).read()) 
            self.idxs = list(self.key_pts.keys())

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])

        self.buffer = 10 # px buffer from border

    def __len__(self):
        return len(self.idxs) # should be the same

    def __getitem__(self, idx):
        key_pts_list = np.array(self.key_pts[self.idxs[idx]])

        length = key_pts_list.shape[-2]

        key_pts_list = key_pts_list.reshape(-1, 2)

        if True:
            # centering
            mean = np.mean(key_pts_list, 0)
            key_pts_list = key_pts_list + (np.array([[224//2, 224//2]]) - mean).astype(int)

        in_img = np.zeros((224, 224))
        output_img = torch.zeros((1, 224, 224))

        for i in range(key_pts_list.shape[0]):
            key_pt = key_pts_list[i]
            key_pts_list[i] = [np.clip(key_pt[0], self.buffer, 223 - self.buffer), np.clip(key_pt[1], self.buffer, 223 - self.buffer)]
            key_pt = key_pts_list[i]
            output_img[0, key_pt[1], key_pt[0]] = 1

        key_pts_list = key_pts_list.reshape(1, -1, 2)

        in_img = cv2.fillPoly(in_img, key_pts_list, color=(255,255,255))

        length_oh = F.one_hot(torch.tensor([length - 1]), MAX_SEQ_LEN) # starts at zero
        key_pts_list = key_pts_list.reshape(length, 2)

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