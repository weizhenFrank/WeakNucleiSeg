
import os
import shutil
import termcolor
import ipdb
import numpy as np
from skimage import morphology, measure
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import distance_transform_edt as dist_tranform
from scipy.ndimage import morphology as ndi_morph
import cv2
import pandas as pd
from skimage.morphology import (erosion, dilation, opening, closing,
                                white_tophat)

import glob
import json
import imageio
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/wliu25/projects/WeakNucleiSeg/')
from network.lib.utils import get_point



def main(partial=0.5, alpha=0.5, box_size=60):
    
    dataset = 'MO'
    partial = partial
    data_dir = f'./data/{dataset}'
    img_dir = f'./data/{dataset}/images'
    label_instance_dir = './data/{:s}/labels_instance'.format(dataset)
    label_point_dir = './data/{:s}/{:.2f}/labels_point'.format(dataset, partial)
    label_cluster_dir = './data/{:s}/{:.2f}/labels_cluster'.format(dataset, partial)
    label_vor_dir = './data/{:s}/{:.2f}/labels_vor'.format(dataset, partial)
    split = '{:s}/train_val_test.json'.format(data_dir)
    stats_path = '{:s}/{:.2f}/stats.csv'.format(data_dir, partial)
    

    with open(split, 'r') as split_file:
        data_list = json.load(split_file)
        img_list = data_list['train'] + data_list['val']
    
    create_point_label(label_instance_dir, label_point_dir, img_list, partial=partial)
                                                                                    
    create_Voronoi_label(label_point_dir, label_vor_dir, img_list)
            
    if not os.path.exists(stats_path):
        if 'MO' in dataset:
            box_size = box_size
        else:
            box_size = 80
            
        Kdist_dir(img_dir, split, stats_path, True, box_size, label_point_dir)
        create_cluster_label(img_dir, label_point_dir, label_vor_dir, label_cluster_dir, img_list, stats_path, alpha=alpha)


def create_point_label(data_dir, save_point_dir, train_list, partial=1):

    print("Generating point label and binary mask from instance label...")
    image_list = os.listdir(data_dir)

    for image_name in tqdm(image_list):
        name = image_name.split('.')[0]
        if '{:s}.png'.format(name[:-6]) not in train_list or name[-5:] != 'label':
            continue

        image_path = os.path.join(data_dir, image_name)
        image = imageio.imread(image_path)
        h, w = image.shape

        # extract bbox
        id_max = np.max(image)
        label_point = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, id_max + 1):
            nucleus = image == i
            if np.sum(nucleus) == 0:
                continue
            x, y = get_point(nucleus)
            if np.random.rand() < partial:
                label_point[x, y] = 255                
        imageio.imwrite('{:s}/{:s}_point.png'.format(save_point_dir, name), label_point.astype(np.uint8))


def create_Voronoi_label(data_dir, save_dir, train_list, postfix='_label_point.png'):
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon
    from network.lib.utils import voronoi_finite_polygons_2d, poly2mask

    print("Generating Voronoi label from point label...")

    for img_name in tqdm(train_list):
        name = img_name.split('.')[0]

        img_path = '{:s}/{:s}{:s}'.format(data_dir, name, postfix)
        label_point = imageio.imread(img_path)
        h, w = label_point.shape

        points = np.argwhere(label_point > 0)
        vor = Voronoi(points)

        regions, vertices = voronoi_finite_polygons_2d(vor)
        box = Polygon([[0, 0], [0, w], [h, w], [h, 0]])
        region_masks = np.zeros((h, w), dtype=np.int16)
        edges = np.zeros((h, w), dtype=np.bool)
        count = 1
        for region in regions:
            polygon = vertices[region]
            # Clipping polygon
            poly = Polygon(polygon)

            poly = poly.intersection(box)  # this is the key
            polygon = np.array([list(p) for p in poly.exterior.coords])
            try:
                mask = poly2mask(polygon[:, 0], polygon[:, 1], (h, w))
            except:
                print(termcolor.colored("error on {:s} image:".format(name), "red"))
                continue
                # ipdb.set_trace()
            edge = mask * (~morphology.erosion(mask, morphology.disk(1)))
            edges += edge
            region_masks[mask] = count
            count += 1

        # fuse Voronoi edge and dilated points
        label_point_dilated = morphology.dilation(label_point, morphology.disk(2))
        label_vor = np.zeros((h, w, 3), dtype=np.uint8)
        label_vor[:, :, 0] = morphology.closing(edges > 0, morphology.disk(1)).astype(np.uint8) * 255
        label_vor[:, :, 1] = (label_point_dilated > 0).astype(np.uint8) * 255

        imageio.imwrite('{:s}/{:s}_label_vor.png'.format(save_dir, name), label_vor)


     
def create_cluster_label(data_dir, label_point_dir, label_vor_dir, save_dir, train_list, stats_path=None, alpha=0.2, radius=20):
        print("Generating cluster label from point label...")
        stats = pd.read_csv(stats_path, index_col=0)

        for img_name in tqdm(train_list):
            name = img_name.split('.')[0]
            scale = stats.loc[name, :][1]

            img = cv2.imread('{:s}/{:s}.png'.format(data_dir, name))
            point = imageio.imread('{:s}/{:s}_label_point.png'.format(label_point_dir, name))
            vor = imageio.imread('{:s}/{:s}_label_vor.png'.format(label_vor_dir, name))

            h, w, _ = img.shape

            if 'MO' in data_dir:
                alpha, radius = alpha, 20
            else:
                alpha, radius = 0.12, 18

            emb = get_emb(img, point, alpha=alpha, scale=scale, radius=radius)

            kmeans = KMeans(n_clusters=3, random_state=0, ).fit(emb)
            clusters = np.reshape(kmeans.labels_, (h, w))
            out = kcluster_post(clusters, vor, point)
            imageio.imwrite('{:s}/{:s}_label_cluster.png'.format(save_dir, name), out)


def get_emb(img, point, alpha=None, scale=None, radius=20):
    dist = dist_tranform(255 - point).reshape(-1, 1)
    dist = np.clip(dist, a_min=0, a_max=radius)
    max_dist = np.max(dist)
    print(max_dist)
    if max_dist != 0:
        dist /= max_dist

    ori_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_image = np.array(ori_image, dtype=np.float).reshape(-1, 3).astype(np.float32) / 255
    print(f"using alpha{alpha}")
    color = (alpha * ori_image / scale).astype(np.float32)

    embeddings = np.concatenate((color, dist), axis=1)

    return embeddings


def kcluster_post(clusters, label_vor, point):
    h, w = point.shape[0:2]
    overlap_nums = [np.sum((clusters == i) * point) for i in range(3)]
    nuclei_idx = np.argmax(overlap_nums)
    remain_indices = np.delete(np.arange(3), nuclei_idx)
    dilated_point = morphology.binary_dilation(point, morphology.disk(5))
    overlap_nums = [np.sum((clusters == i) * dilated_point) for i in remain_indices]
    background_idx = remain_indices[np.argmin(overlap_nums)]

    nuclei_cluster = clusters == nuclei_idx
    background_cluster = clusters == background_idx

    # refine clustering results
    nuclei_labeled = measure.label(nuclei_cluster)
    initial_nuclei = morphology.remove_small_objects(nuclei_labeled, 30)
    refined_nuclei = np.zeros(initial_nuclei.shape, dtype=bool)

    voronoi_cells = measure.label(label_vor[:, :, 0] == 0)
    voronoi_cells = morphology.dilation(voronoi_cells, morphology.disk(2))

    unique_vals = np.unique(voronoi_cells)
    cell_indices = unique_vals[unique_vals != 0]
    N = len(cell_indices)
    for i in range(N):
        cell_i = voronoi_cells == cell_indices[i]
        nucleus_i = cell_i * initial_nuclei

        nucleus_i_dilated = morphology.binary_dilation(nucleus_i, morphology.disk(5))
        nucleus_i_dilated_filled = ndi_morph.binary_fill_holes(nucleus_i_dilated)
        nucleus_i_final = morphology.binary_erosion(nucleus_i_dilated_filled, morphology.disk(7))
        refined_nuclei += nucleus_i_final > 0

    refined_label = np.zeros((h, w, 3), dtype=np.uint8)
    point_dilated = morphology.dilation(point, morphology.disk(10))
    refined_label[:, :, 0] = (background_cluster * (refined_nuclei == 0) * (point_dilated == 0)).astype(
        np.uint8) * 255
    refined_label[:, :, 1] = refined_nuclei.astype(np.uint8) * 255

    return refined_label


def Kdist(emb, size=9, use_local=False, guide=None):
    print("calculating the std of color distance label from point label...")

    if len(emb.shape) > 3:
        raise Exception("dim can't be larger than 3")

    if len(emb.shape) == 2:
        emb = emb[..., np.newaxis]

    H, W, C = emb.shape

    rad = size // 2
    dists = []

    for r in range(H):
        br = r - rad if r - rad > 0 else 0
        er = r + rad if r + rad < H else H - 1

        for c in range(W):
            bc = c - rad if c - rad > 0 else 0
            ec = c + rad if c + rad < W else W - 1
            region = emb[br:er + 1, bc:ec + 1, ...]
            if not use_local:
                if C == 1 and len(set(region.reshape(-1))) == 1:
                    continue
                index_rc = (ec + 1 - bc) * (r - br) + (c - bc)
                p_h, p_w, _ = region.shape
                region = region.reshape(p_h * p_w, C)
                if not np.all(region[index_rc] == emb[r, c, ...]):
                    ipdb.set_trace()
                region = np.delete(region, index_rc, 0)
                dist = np.linalg.norm(emb[r, c, ...] - region, axis=-1)
            else:
                dist = np.linalg.norm(emb[r, c, ...] - region, axis=-1)
                local_guide = guide[br:er + 1, bc:ec + 1] > 0
                local_guide[rad, rad] = False
                dist = dist[local_guide]
            dists.append(dist.ravel())
    dists = np.concatenate(dists)
    return dists


def Kdist_dir(img_dir, split, save_path=None, use_local=False, box_size=None, point_dir=None):
    print("Calculating std...It might take a while")
    split = read_split(split, "train")

    stats = dict()

    for idx, i in enumerate(split):
        img = cv2.imread(os.path.join(img_dir, f"{i}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color_emb = np.array(img, dtype=float) / 255

        if use_local:
            point = imageio.imread(os.path.join(point_dir, f"{i}_label_point.png"))
            stru = np.ones((box_size, box_size))
            guide = dilation(point, stru)
            dists = Kdist(color_emb, 9, True, guide)
        else:
            dists = Kdist(color_emb, 9)

        mean = dists.mean()
        std = dists.std()
        stats[i] = {'color_dist_mean': mean, 'color_dist_std': std}

    stats = pd.DataFrame.from_dict(stats, orient="index").round(4)
    stats.to_csv(save_path)


def read_split(split, img_set):
    with open(split, "r") as f:
        split = json.load(f)[img_set]
    split = [i.strip(".png") for i in split]
    return split


def creat_inslabel_from_se(split_file, split, se_dir, epoch, dst_dir):
    split_set = read_split(split_file, split)
    source_dir = f"{se_dir}/{epoch}/{split}_segmentation"
    os.makedirs(dst_dir, exist_ok=True)

    for name in split_set:
        img_path = f"{source_dir}/{name}_seg_prob.tiff"
        dst = f"{dst_dir}/{name}_label_prob.tiff"
        print(f"copy from {img_path} to {dst}")
        shutil.copyfile(img_path, dst)


def cal_metrics(img_path, label_path):
    from sklearn.metrics import precision_recall_fscore_support as prf
    from sklearn.metrics import classification_report as cr
    img_path = "/home/wliu25/projects/WeakNucleiSeg/data/MO/1.00/labels_cluster/Breast_TCGA-A7-A13F-01Z-00-DX1_label_cluster.png"
    label_path = "/home/wliu25/projects/WeakNucleiSeg/data/MO/1.00/labels_binary/Breast_TCGA-A7-A13F-01Z-00-DX1_label_binary.png"
    img = imageio.imread(img_path)
    label = imageio.imread(label_path)

    bin = (np.ones(img.shape[:2])*2).astype(np.uint8)
    bin[img[:,:, 0]==255] = 0
    bin[img[:,:, 1]==255] = 1

    result = cr(label.reshape(-1), bin.reshape(-1), target_names=['bg', 'nuclei', 'unknown'])
    with open("debug_metric.txt", "w") as f:
        f.write(result)

    
    
    
    
    
if __name__ == "__main__":
    cal_metrics()
    