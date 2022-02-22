import copy
import os.path

import numpy as np
import torch
from skimage import measure
from tqdm import tqdm

from network.lib.utils.utils import cca, cluster, softmax


def padding_img(input_img, patch_size, overlap):
    if not isinstance(input_img, torch.Tensor):
        img = torch.from_numpy(input_img)
    else:
        img = input_img
    b = c = 1
    img_dim = len(img.size())

    if img_dim == 2:
        h0, w0 = img.size()
    elif img_dim == 3:
        c, h0, w0 = img.size()
    elif img_dim == 4:
        b, c, h0, w0 = img.size()
    else:
        raise ValueError("image size must be less than or equal 4")
    img = img.view(b, c, h0, w0)
    # zero pad for border patches
    pad_h = 0
    # here the padding is to make the the image size could be divided exactly by the size - overlap (the length of step)
    if h0 - patch_size > 0 and (h0 - patch_size) % (patch_size - overlap) > 0:
        pad_h = (patch_size - overlap) - (h0 - patch_size) % (patch_size - overlap)  # size is the the input size of model
        tmp = torch.zeros((b, c, pad_h, w0))
        img = torch.cat((img, tmp), dim=2)

    if w0 - patch_size > 0 and (w0 - patch_size) % (patch_size - overlap) > 0:  # same as the above
        pad_w = (patch_size - overlap) - (w0 - patch_size) % (patch_size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        img = torch.cat((img, tmp), dim=3)

    if img_dim == 2:
        img = img[0, 0, :, :]
    elif img_dim == 3:
        img = img[0, :, :, :]
    else:
        img = img

    if not isinstance(input_img, torch.Tensor):
        return img.numpy()
    else:
        return img


def get_probmaps(inputs, model, opt):
    output = split_forward(model, inputs, opt)
    output = output.squeeze(0).cpu().numpy()
    output = softmax(output)
    return output


def split_forward(model, input, opt):
    """
    split the input image for forward process
    """

    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    # here the padding is to make the the image size could be divided exactly by the size - overlap (the length of step)
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)  # size is the the input size of model
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:  # same as the above
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), 2, h, w))

    for i in range(0, h - overlap, size - overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h

        for j in range(0, w - overlap, size - overlap):
            c_end = j + size if j + size < w else w

            ind2_s = j + overlap // 2 if j > 0 else 0
            ind2_e = j + size - overlap // 2 if j + size < w else w

            input_patch = input[:, :, i:r_end, j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch = model(input_var)
                output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i,
                                                             ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0].cuda()
    return output


def loc_guide(labeled_img, idxes):
    loc = np.zeros_like(labeled_img, dtype=bool)

    for i in idxes:
        loc[labeled_img == i] = True

    return loc


def get_overlap_inline(region, indexs):
    overlap_region = copy.deepcopy(region)
    nucs = np.unique(overlap_region)[1:]
    for i in nucs:
        if i not in indexs:
            overlap_region[overlap_region == i] = 0

    return overlap_region > 0


def instance_inference(model, seg_model, img, point, opt, img_name=None, se_fg=None):

    os.makedirs(opt.intermediate_dir, exist_ok=True)
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    b, c, h0, w0 = img.size()

    # zero pad for border patches
    pad_h = 0
    # here the padding is to make the the image size could be divided exactly by the size - overlap (the length of step)
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)  # size is the the input size of model
        tmp = torch.zeros((b, c, pad_h, w0))
        img = torch.cat((img, tmp), dim=2)
        point = np.concatenate((point, np.zeros((pad_h, w0))), axis=0)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:  # same as the above
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        img = torch.cat((img, tmp), dim=3)
        point = np.concatenate((point, np.zeros((h0 + pad_h, pad_w))), axis=1)

    if se_fg is not None:
        se_fg = padding_img(se_fg, size, overlap)

    _, c, h, w = img.size()
    output = np.zeros((h, w))

    for i in tqdm(range(0, h - overlap, size - overlap)):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        # the new ind1_s equals the last ind1_e as (i+1) + overlap // 2 - (i) - size + overlap // 2 = 0.
        for j in tqdm(range(0, w - overlap, size - overlap)):
            c_end = j + size if j + size < w else w
            ind2_s = j + overlap // 2 if j > 0 else 0
            ind2_e = j + size - overlap // 2 if j + size < w else w

            input_patch = img[:, :, i:r_end, j:c_end]
            point_patch = point[i:r_end, j:c_end]
            input_var = input_patch.cuda()
            base = output.max().item()

            with torch.no_grad():

                output_patch = model(input_var).squeeze(0).cpu().numpy()
                if se_fg is not None:
                    foreground_patch = se_fg[i:r_end, j:c_end]
                    foreground_patch = cca(foreground_patch, opt, fg_prob=True)
                else:
                    seg_output_patch = seg_model(input_var).squeeze(0).cpu().numpy()
                    foreground_patch = cca(seg_output_patch, opt, fg_prob=False)

                if (foreground_patch > 0).sum() == 0:
                    instance_seg_patch = foreground_patch
                else:
                    instance_seg_patch = cluster(embedding=output_patch, foreground=foreground_patch, point=point_patch,
                                                 method=opt.post.cluster_method, base=base,
                                                 bandwidth=opt.post.bandwidth)

                if opt.post.ins_post:
                    instance_seg_patch, _ = patch_post(instance_seg_patch, 5)
                    # patch_save_path = os.path.join(opt.intermediate_dir, f"{img_name}_{i}_{j}_post.tiff")
                    # imageio.imwrite(patch_save_path, instance_seg_patch)
                # hor
                # get the nuc who go through the hor line in the mean shift patch
                # first we labeled to avoid selecting multiple nuclei with same mean-shift label

                instance_seg_patch_temp_labeled = measure.label(instance_seg_patch)
                # here we only focus on the splicing region in patch
                nuc_in_hor_line = np.unique(instance_seg_patch_temp_labeled[ind1_s - i, ind2_s - j:ind2_e - j])[1:]

                overlap_in_hor_line_loc = get_overlap_inline(
                    instance_seg_patch_temp_labeled[:ind1_s - i, :], nuc_in_hor_line)

                overlap_in_hor_line = instance_seg_patch[:ind1_s - i, :]

                output[i:ind1_s, j:c_end][overlap_in_hor_line_loc] = overlap_in_hor_line[overlap_in_hor_line_loc]

                # ver
                # get the nuc who go through the ver line in the mean shift patch
                # first we labeled to avoiding select multiple nuclei with same mean-shift label
                nuc_in_ver_line = np.unique(instance_seg_patch_temp_labeled[ind1_s - i:ind1_e - i, ind2_s - j])[1:]

                # get the overlap region
                overlap_in_ver_line_loc = get_overlap_inline(
                    instance_seg_patch_temp_labeled[:, :ind2_s - j], nuc_in_ver_line)
                overlap_in_ver_line = instance_seg_patch[:, :ind2_s - j]
                # only keep the nuc in the line in overlap region and put the nuc in the big output

                output[i:r_end, j:ind2_s][overlap_in_ver_line_loc] = overlap_in_ver_line[overlap_in_ver_line_loc]

                # splicing the patches
                output[ind1_s:ind1_e, ind2_s:ind2_e] = instance_seg_patch[ind1_s - i:ind1_e - i, ind2_s - j:ind2_e - j]

    output = output[:h0, :w0].astype(np.int32)

    return output


def patch_post(patch, num_region=5):
    nuc_list = list(np.unique(patch))
    nuc_list.pop(0)
    num_nuc_dict = {}

    for nuc in nuc_list:
        nuc_map = measure.label(patch == nuc)
        num_nuc = len(np.unique(nuc_map)[1:])
        num_nuc_dict[nuc] = num_nuc
    sort_orders = sorted(num_nuc_dict.items(), key=lambda x: x[1], reverse=True)
    keep_nuc = {}

    for k, v in sort_orders:
        if v >= num_region:
            keep_nuc[k] = v

    if len(keep_nuc) > 0:
        for idx, (k, v) in enumerate(keep_nuc.items()):
            if idx > 0:
                break
            patch[patch == k] = 0

    return patch, sort_orders


