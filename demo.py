import torch 
import numpy as np 
from tqdm import trange, tqdm
import matplotlib.pyplot as plt 
import cv2
from scipy.spatial import ConvexHull
import alphashape
from descartes import PolygonPatch


from datasets import COCOPolygonDataset
from helpers import angle_between, euclid_dis, unit_vector

def regenerate_polygon(true_mask, nodes_list, node_coords, pred_pos):
    # path = findLongestPath(nodes_list, node_coords, true_mask)
    # polygon = ConvexHull(node_coords.reshape(-1, 2)).vertices
    points = node_coords.reshape(-1, 2)
    # alpha = 0.95 * alphashape.optimizealpha(points)
    alpha = 0
    hull = alphashape.alphashape(points, alpha)

    img = np.zeros((224, 224))

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = np.array([int_coords(hull.exterior.coords)])

    for i in range(exterior.shape[1]):
        x, y = exterior[0, i]
        exterior[0, i] = [np.clip(y, 0, 223), np.clip(x, 0, 223)]

    img = cv2.fillPoly(img, exterior, color=(255,255,255))

    for interior in hull.interiors:
        for i in range(interior.shape[1]):
            x, y = interior[0, i]
            interior[0, i] = [np.clip(y, 0, 223), np.clip(x, 0, 223)]

        img = cv2.fillPoly(img, interior, color=(0,0,0))


    return img

def main():
    device = torch.device("cpu")

    model = torch.load("PolygonPredictor.pt").to(device)

    val_dataset = COCOPolygonDataset('key_pts/key_pts_instances_val2017.json')

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size = 16,
                                        num_workers = 0,
                                        shuffle = True)

    for in_imgs, true_len, true_pos in tqdm(val_loader, unit='batch'):
        in_imgs = in_imgs.to(device=device, dtype=torch.float32)
        true_len = true_len.to(device, dtype=torch.float32).squeeze(1)

        pred_len, pred_pos = model(in_imgs)

        for i in range(in_imgs.shape[0]):
            plt.figure(figsize=(15,10))
            ##################
            plt.subplot(3,2,1)
            plt.title("Mask In")
            mask_in = in_imgs[i][0].cpu().detach()
            plt.imshow(mask_in)

            ##################
            plt.subplot(3,2,2)
            p_l = torch.argmax(pred_len[i]) + 1
            p = pred_pos[i][0].cpu().detach()
            plt.title(f"predict {p_l} vertices. range: [{p.min():.1f}, {p.max():.1f}]")
            p = (p - p.min()) / (p.max() - p.min())
            pred_p = p
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
            node_coords = indices
            topk_image = np.zeros((224,224), dtype=float)
            for index in indices:
                topk_image[index[0], index[1]] = 1

            plt.imshow(topk_image)

            #################
            plt.subplot(3,2,6)
            plt.title("regenerated convex-hull from prediction")
            regenerated = regenerate_polygon(mask_in, [i for i in range(p_l)], node_coords, pred_p)
            plt.imshow(regenerated)

            plt.show()

if __name__ == "__main__":
    main()