
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass

import imageio
import numpy as np
import pandas as pd
import torch
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial import Voronoi
from skimage import draw
from skimage import measure, morphology
from skimage.morphology import erosion, disk


def mk_colored(instance_img):
    instance_img = instance_img.astype(np.int32)
    H, W = instance_img.shape[0], instance_img.shape[1]
    pred_colored_instance = np.zeros((H, W, 3))

    nuc_index = list(np.unique(instance_img))
    nuc_index.pop(0)

    for k in nuc_index:
        pred_colored_instance[instance_img == k, :] = np.array(get_random_color())

    return pred_colored_instance


def get_centroid(img):
    img = measure.label(img)
    H, W = img.shape[0], img.shape[1]
    pred_regions = measure.regionprops(img)
    pred_points = []
    for region in pred_regions:
        pred_points.append(region.centroid)
    pred_points = np.array(pred_points, dtype=np.int64)
    pred = np.zeros((H, W))
    pred[pred_points[:, 0], pred_points[:, 1]] = 255
    return pred


def get_point(img):
    a = np.where(img != 0)
    rmin, rmax, cmin, cmax = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return (rmin + rmax) // 2, (cmin + cmax) // 2


def assign_point(image_instance):
    id_max = np.max(image_instance)
    img_shape = image_instance.shape
    label_point = np.zeros(img_shape, dtype=np.uint8)

    for i in range(1, id_max + 1):
        nucleus = image_instance == i
        if np.sum(nucleus) == 0:
            continue
        x, y = get_point(nucleus)
        label_point[x, y] = 255

    return label_point


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


# borrowed from https://gist.github.com/pv/8036995
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b


def show_figures(imgs: list, opt=None, image_name=None, grid=None, postfix=""):
    if not os.path.exists(opt.intermediate_dir):
        os.makedirs(opt.intermediate_dir)

    import matplotlib.pyplot as plt

    if grid is not None:
        row = grid[0]
        col = grid[1]
    else:
        row = 2
        col = int(len(imgs) / 2)
    fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(10 * col, 10 * row))
    axs = axs.flatten()
    for idx, img in enumerate(imgs):
        im = axs[idx].imshow(img)
        fig.colorbar(im, ax=axs[idx])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    img_path = os.path.join(opt.intermediate_dir,
                            "{:s}_{:s}.png".format(str(image_name), str(postfix)))
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()


def onehot_encoding(x):
    """ from HxW to CxHxW one-hot encoding"""
    C = x.max() + 1
    out = np.zeros((C, x.size), dtype=np.uint8)
    out[x.ravel(), np.arange(x.size)] = 1
    out.shape = (C,) + x.shape
    return out


# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, shape=1):
        self.shape = shape
        self.reset()
        self.total = {}

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def record(self, val, key):
        self.total[key] = val


def write_txt(results, filename, mode='w'):
    """ Save the result of losses and F1 scores for each epoch/iteration
        results: a list of numbers
    """
    with open(filename, mode) as file:
        num = len(results)
        for i in range(num - 1):
            file.write('{:.4f}\t'.format(results[i]))
        file.write('{:.4f}\n'.format(results[num - 1]))


def save_results(header, all_result, test_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average results:\n')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(all_result[i]))
        file.write('{:.4f}\n'.format(all_result[N - 1]))
        file.write('\n')

        # results for each image
        for key, vals in sorted(test_results.items()):
            file.write('{:s}:\n'.format(key))
            for value in vals:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')


def show_params_status(model):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    for p_name, p in model.named_parameters():

        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        strings.append("{:<60s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))
    strings = '\n'.join(strings)
    return f"\n{strings}\n ----- \n \n{total_params / 1000000.0:.3f}M total parameters \n "


def save_config(conf_dict, save_dir):
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        conf_s = json.dumps(conf_dict, indent=3)
        f.write(conf_s)
    return conf_s


def get_fold_report(data_dir, pattern_1, pattern_2, file_name, num_folds, save_dir):
    pattern_str = os.path.join(data_dir, pattern_1, pattern_2, file_name)
    target_dir = os.path.join(save_dir, pattern_2.strip("*"))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    files = glob.glob(pattern_str)
    files.sort()
    metric_value = {'train': defaultdict(list), 'val': defaultdict(list)}
    assert num_folds == len(files)
    for file in files:
        with open(file) as f:
            file_list = [line.rstrip('\n') for line in f]
            # ipdb.set_trace()
            if not (('train' in file_list[-2].split('\t')[0]) and ('val' in file_list[-1].split('\t')[0])):
                raise Exception("no best score!")
            values = dict()
            values['train'] = [metric.split(':')[-2:] for metric in file_list[-2].split('\t')]
            values['val'] = [metric.split(':')[-2:] for metric in file_list[-1].split('\t')]
            for split, scores in values.items():
                # ipdb.set_trace()
                for v in scores:
                    metric_value[split][v[0].strip()].append(float(v[1]))
    # ipdb.set_trace()
    for i in ['train', 'val']:
        final_score = pd.DataFrame(metric_value[i])
        final_score = final_score.append(final_score.describe())
        final_score = final_score.round(4)
        # ipdb.set_trace()
        final_score.to_csv(os.path.join(target_dir, "{:s}_best_folds.csv".format(i)), index_label=False, index=True)


def generate_hard_label(prob_label, cutoff, remove_area=None):
    assert cutoff[1] >= cutoff[0]
    # generate hard pseudo label according to the prob and cutoff
    pseudo_label = torch.ones_like(prob_label) * 2  # unlabeled region has index 2
    pseudo_label[prob_label < cutoff[0]] = 0  # background region has index 0
    pseudo_label[prob_label >= cutoff[1]] = 1  # foreground region has index 1
    if remove_area is not None:
        pseudo_label = pseudo_label.numpy().astype(np.int64)
        pred_labeled = []

        for i in pseudo_label:
            i_labeld = measure.label(i)
            i_labled_re = morphology.remove_small_objects(i_labeld, remove_area)
            i_labled_re_filled = binary_fill_holes(i_labled_re > 0)
            pred_labeled.append(i_labled_re_filled)
        pred_labeled = np.stack(pred_labeled, axis=0)
        return pred_labeled
    else:
        return pseudo_label.long().cpu().numpy()


def get_bg(prob, thr1=0, thr2=0.01):
    bg = torch.zeros_like(prob)
    bg[(prob > thr1)*(prob <= thr2)] = 1
    return bg.numpy().astype(np.int64)


def generate_weights_nllloss_2d(prob_label, hard_label):
    prob_foreground_weight = torch.clone(prob_label)
    prob_background_weight = torch.ones_like(prob_foreground_weight) - prob_foreground_weight
    prob_weight = torch.stack(
        (prob_background_weight, prob_foreground_weight, torch.zeros_like(prob_background_weight)), dim=1)

    indexed_prob_weight = torch.gather(prob_weight, 1, hard_label.unsqueeze(1)).squeeze(1).cuda()
    # select the pixel weight by hard pseudo label
    return indexed_prob_weight


def generate_soft_label(prob_label):
    prob_foreground = torch.clone(prob_label)
    prob_background = torch.ones_like(prob_foreground) - prob_foreground
    soft_label = torch.stack((prob_background, prob_foreground), dim=1)  # order matters!!

    return soft_label


@dataclass
class RemoveKNoise:
    split_path: str
    k_label_dir: str
    label_dir: str
    prob_dir: str
    save_dir: str

    def __post_init__(self):
        self.make_dir()
        print("begin initialization")

    def make_dir(self, *args):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if len(args) != 0:
            for target_dir in args:
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

    def read_split(self, set_name):
        with open(self.split_path) as f:
            train_val_test = json.load(f)
        set_list = train_val_test[set_name]
        return set_list

    def read_labels(self, img_name):
        k_label = imageio.imread(os.path.join(self.k_label_dir, "{:s}_label_cluster.png".format(img_name)))
        label_path = os.path.join(self.label_dir, "{:s}_label.png".format(img_name))
        if not os.path.exists(label_path):
            label_path = os.path.join(self.label_dir, "{:s}_pred.png".format(img_name))
        label = imageio.imread(label_path)
        return k_label, label

    def remove_noise_k_label_by_gt(self):
        train_list = self.read_split('train')

        for img in train_list:
            img_name = os.path.splitext(img)[0]
            k_label, label = self.read_labels(img_name)

            new_label = np.zeros_like(k_label)
            label = label > 0
            new_label[:, :, 0][label == 0] = 255
            new_label[:, :, 1][label == 1] = 255
            new_k_cluster = np.where(new_label == k_label, k_label, 0)
            imageio.imwrite(os.path.join(self.save_dir, "{:s}_label_cluster.png".format(img_name)), new_k_cluster)

    def remove_noise_k_label_by_erosion(self):
        train_list = self.read_split('train')
        for img in train_list:
            img_name = os.path.splitext(img)[0]
            k_label, label = self.read_labels(img_name)

            foreground = erosion(k_label[:, :, 1], disk(3))
            background = erosion(k_label[:, :, 0], disk(3))

            new_k_cluster = np.zeros_like(k_label)
            new_k_cluster[:, :, 0] = background
            new_k_cluster[:, :, 1] = foreground
            imageio.imwrite(os.path.join(self.save_dir, "{:s}_label_cluster.png".format(img_name)), new_k_cluster)

    def remove_by_prob(self, cutoffs):

        train_list = self.read_split('train')

        for idx, img in enumerate(train_list):
            img_name = os.path.splitext(img)[0]
            probmap = imageio.imread(os.path.join(self.prob_dir, "{:s}_prob.tiff".format(img_name)))

            image_shape = list(probmap.shape)
            image_shape.append(3)
            for cutoff in cutoffs:
                cut_label = np.zeros(image_shape)
                cut_label[:, :, 0] = np.where(probmap < (1 - cutoff), 255, 0)
                cut_label[:, :, 1] = np.where(probmap > cutoff, 255, 0)

                sub_dir = os.path.join(self.save_dir, "cutoff_{:.2f}".format(cutoff))
                self.make_dir(sub_dir)
                imageio.imwrite(os.path.join(sub_dir, "{:s}_label_pseudo.png".format(img_name)), cut_label)


def point_reg_metric(pred, gt, cutoff=10):
    from scipy.spatial.distance import cdist

    pred = pred > 0

    pred_idx = measure.label(pred)
    gt_idx = measure.label(gt)
    gt_points = np.array(np.where(gt_idx > 0)).T
    pred_points = np.array(np.where(pred_idx > 0)).T

    dist_mat = cdist(gt_points, pred_points)
    gt_points_num, pred_points_num = dist_mat.shape
    while (0 not in dist_mat.shape) and (np.min(dist_mat) <= cutoff):
        x, y = np.unravel_index(dist_mat.argmin(), dist_mat.shape)
        del_row = np.delete(dist_mat, x, 0)
        del_col = np.delete(del_row, y, 1)
        dist_mat = del_col

    rest_gt_points_num, rest_pred_points_num = dist_mat.shape

    hit = (gt_points_num - rest_gt_points_num)
    recall = hit / (gt_points_num + 1e-8)
    precision = hit / (pred_points_num + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return recall, precision, f1


def check_point_gauss(maps):
    len_map = maps.shape[0]
    maps = [maps[i, :, :] for i in range(len_map)]
    ins_maps = [measure.label(maps[i] > 0) for i in range(len_map)]
    max_nuclei_values_maps = {}
    area_nuclei_values_maps = {}
    for idx, ins_map in enumerate(ins_maps):
        max_nuclei_values = []
        area_each_nuclei = []
        for i in range(1, np.max(ins_map) + 1):
            area_each_nuclei.append(torch.sum((maps[idx] > 0)[ins_map == i]))
            max_nuclei_values.append(torch.max(maps[idx][ins_map == i]))
        max_nuclei_values_maps[idx] = max_nuclei_values
        area_nuclei_values_maps[idx] = area_each_nuclei
    return max_nuclei_values_maps, area_nuclei_values_maps


def cluster(embedding, foreground, point, method="Kmeans", base=0, bandwidth=1.0, cluster_all=True):
    """
    :param base:
    :param embedding:  The numpy array shape is C * H * W
    :param foreground: The numpy array shape is H * W
    :param point: The shape numpy array is H * W
    :return: The numpy array instance labeled image

    """

    from sklearn.cluster import KMeans, MeanShift
    embedding = embedding.transpose(1, 2, 0)
    H, W, c = embedding.shape
    embedding = embedding.reshape(-1, c)
    foreground = foreground.reshape(H * W).astype(np.int32)

    if method == 'Kmeans':
        if point.sum() == 0:
            return np.zeros((H, W))
        point = point.reshape(H * W).astype(np.int32)
        cluster_center = embedding[point > 0]

        embedding = embedding[foreground > 0]
        label = KMeans(n_clusters=cluster_center.shape[0], init=cluster_center, max_iter=500).fit_predict(embedding)
    elif method == 'mean_shift':
        embedding = embedding[foreground > 0]
        label = MeanShift(bandwidth=bandwidth, n_jobs=-1, cluster_all=cluster_all).fit_predict(embedding)
    else:
        raise Exception("Give valid method")

    label[label != -1] += 1
    foreground[foreground > 0] = label + base

    foreground = foreground.reshape(H, W)

    return foreground


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    ex = np.exp(x - np.max(x, axis=0))
    return ex / np.sum(ex, axis=0)


def cca(pred, opt, fg_prob=False):
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    cutoff = opt.post['cutoff']
    if not fg_prob:
        pred = softmax(pred)
        pred = pred[1, :, :]
    pred[pred <= cutoff] = 0
    pred[pred > cutoff] = 1
    pred = pred.astype(int)

    pred_labeled = measure.label(pred)
    pred_labeled = morphology.remove_small_objects(pred_labeled, opt.post['min_area'])
    pred_labeled = binary_fill_holes(pred_labeled > 0)
    pred_labeled = measure.label(pred_labeled)

    return pred_labeled


def save_checkpoint(state, epoch, save_dir, is_best, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))


def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.4f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def get_increased_FP(pred, thr, gt):
    if isinstance(pred, np.ndarray):
        pred = np.array(pred)

    pred_map = softmax(pred)
    pred_map = pred_map[1, :, :]
    pred = (pred_map >= thr).astype(int)
    pred = semantic_post(pred)

    base = (pred_map >= 0.5).astype(int)
    base = semantic_post(base)

    bg_residual = pred - base
    false_positive = (bg_residual == 1).astype(int) * (gt == 0).astype(int)

    return false_positive


def semantic_post(pred):

    pred_labeled = measure.label(pred)
    pred_labeled = morphology.remove_small_objects(pred_labeled, 20)
    pred = binary_fill_holes(pred_labeled > 0).astype(int)
    return pred


def copydir(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


if __name__ == "__main__":
    import glob

    pass
