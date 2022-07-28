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
    # filename = "annotations/instances_train2017.json" # should be imput later
    filename = "annotations/instances_val2017.json" # should be imput later


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

        # if len(pts) > MAX_SEQ_LEN:
        #     pts = simplify_pts(pts)

        #######################################################################################
        # centering
        mean = np.mean(pts, 0)
        print(f"pts shape {pts.shape}")
        pts = pts + (np.array([[224//2, 224//2]]) - mean)
        # pts = pts - mean + 

        # rescaling
        # rescale_range = 112
        # min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        # range_x = max_x - min_x
        # min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        # range_y = max_y - min_y

        # if min(range_x, range_y) == 0:
        #     pts = np.array([[[100, 100], [100, 200], [200, 200], [200, 100], [100, 100]]])
        #     length = 5
        # elif min(range_x, range_y) < 50:
        #     r_mul = 200 / (max(range_x, range_y) + 1) # plus one to avoid zero division

        #     pts[:, 0] = (r_mul * pts[:, 0].astype(np.float32)).astype(np.int8) 
        #     pts[:, 1] = (r_mul * pts[:, 1].astype(np.float32)).astype(np.int8)

        #     # centering
        #     mean = np.mean(pts, 0)
        #     pts = pts + (np.array([[224//2, 224//2]]) - mean)
        ######################################################################################

        pts_simplified = simplify_pts(pts)
        pts_simplified = np.int32([pts_simplified]) 
        pts = np.int32([pts])
        # print(pts)

        key_pts[annot['id']] = pts.tolist()

        # img_mat = np.zeros((img['height'], img['width']))
        img_mat = np.zeros((224,224,3))
        img_mat = cv2.fillPoly(img_mat, pts, color=(255,255,255))

        for pt in pts[0]:
            cv2.circle(img_mat, [pt[0], pt[1]], 1, (255, 0, 0))

        img_mat_simple = np.zeros((224,224,3))
        img_mat_simple = cv2.fillPoly(img_mat, pts_simplified, color=(255,255,255))

        plt.subplot(1, 2, 1)
        plt.imshow(img_mat)
        plt.title(f"mask of {labels[annot['category_id'] - 1]}, {pts_simplified.shape[-2]} vertices")
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_mat_simple)
        plt.title(f"simplified {pts_simplified.shape[-2]} vertices")
        plt.show()

        # cv2.imwrite(annot_path_name, img_mat)

        # if i % 100 == 0:
        #     with open(json_pts_path, "w") as f:
        #         f.write(json.dumps(key_pts))