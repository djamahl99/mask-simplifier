import json
import os
from unittest import expectedFailure
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 
import rdp

MAX_SEQ_LEN = 50

def simplify_pts(pts):
    delta = 1e-1
    epsilon = delta
    its = 0

    # print(pts.shape)

    if len(pts.shape) == 3:
        # print("shape", pts.shape)
        pts = pts[0]

    while pts.shape[0] > MAX_SEQ_LEN:
        pts = rdp.rdp(pts, epsilon)

        epsilon += delta 
        its += 1

    # print(f"took {its} iterations and epsilon = {epsilon}")
    return np.array([pts], dtype=np.int32)

if __name__ == "__main__":
    filename = "annotations/instances_train2017.json" # should be imput later


    labels_txt = open("annotations/labels.txt").read()
    labels = labels_txt.splitlines()

    mask_folder_path = "masks/" + os.path.basename(filename).replace('.json', '')
    json_pts_path = "key_pts/key_pts_" + os.path.basename(filename)

    print("mask folder path", mask_folder_path)

    j = json.loads(open(filename).read())

    img_id_to_img = {img['id']: img for img in j['images']}
    # print(j['images'][0])
    # exit()

    category_stats = {}

    annots = j['annotations']

    if not os.path.exists(mask_folder_path):
        os.makedirs(mask_folder_path)

    new_shape = (224, 224)

    key_pts = {}

    i = 0

    for annot in tqdm(annots, desc='Generating masks', unit='masks'):
        i += 1
        annot_path_name = f"{mask_folder_path}/{annot['id']}.jpg"

        # if os.path.exists(annot_path_name):
        #     continue
        #     raise Exception("Path already exists - " + annot_path_name)

        img = img_id_to_img[annot['image_id']]

        w, h = img['width'], img['height']
        n_w, n_h = (w / max(w,h)) * new_shape[0], (h / max(w,h)) * new_shape[1]
        r_w, r_h = n_w / w, n_h / h

        # print("new ", (n_w, n_h), "old", (w, h), "r", (r_w, r_h))

        try:
            pts = np.array([[annot['segmentation'][0][2*i] * r_w, annot['segmentation'][0][2*i+1] * r_h] for i in range(len(annot['segmentation'][0]) // 2)])
        except Exception as e:
            continue

        if len(pts) > MAX_SEQ_LEN:
            pts = simplify_pts(pts)

        pts = np.int32([pts])
        # print(pts)

        key_pts[annot['id']] = pts.tolist()

        # img_mat = np.zeros((img['height'], img['width']))
        img_mat = np.zeros(new_shape)

        img_mat = cv2.fillPoly(img_mat, pts, color=(255,255,255))


        cv2.imwrite(annot_path_name, img_mat)

        # plt.imshow(img_mat)
        # plt.title(labels[annot['category_id'] - 1])
        # plt.show()

        if i % 100 == 0:
            with open(json_pts_path, "w") as f:
                f.write(json.dumps(key_pts))