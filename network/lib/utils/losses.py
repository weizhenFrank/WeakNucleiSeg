import copy

import numpy as np
import skimage.measure as measure
import torch
import torch.nn as nn
import torch.nn.functional as F


def full_ins_emb(embedding, foreground, bg=None):
    """
    This function only differs the same name function in utils on the empty region.
    Here we we skip, and there we use the zero as embedding.
    """
    bz = embedding.shape[0]
    batch_instance_emb = []
    ins_labels = copy.deepcopy(foreground)
    for i in range(bz):

        pred = embedding[i]
        pred = pred.permute(1, 2, 0)
        label = ins_labels[i]

        instance_label = measure.label(label)
        instances_embedding = []

        for idx in range(1, instance_label.max() + 1):
            instance = instance_label == idx
            if instance.sum() > 9:  # filter the nuclei noise whose size is smaller than 3 * 3
                instances_embedding.append(pred[instance])
        if len(instances_embedding) != 0:
            if bg is not None:
                instances_embedding.append(pred[bg[i] > 0])
            batch_instance_emb.append(instances_embedding)
    return batch_instance_emb


def get_instance_emb(embedding, foregrounds, labels, bg=None):
    bz = embedding.shape[0]
    batch_instance_emb = []
    for i in range(bz):

        pred = embedding[i]
        pred = pred.permute(1, 2, 0)
        foreground = foregrounds[i]
        label = labels[i]
        point_label = np.zeros_like(label)
        point_label[label == 1] = 1

        label[label == 1] = 2  # only keep vor edge
        instance_label = measure.label(label, connectivity=1)

        instances_embedding = []

        for idx in range(1, instance_label.max() + 1):
            instance = instance_label == idx

            instance_foreground = foreground * instance

            if np.sum(instance_foreground) > 0:

                if np.sum(instance_foreground * point_label) > 0:  # to tell whether if the polygon contains a point
                    polygon_nuc = measure.label(instance_foreground)
                    assert polygon_nuc.max() != 0
                    if polygon_nuc.max() > 1:  # to tell whether if the foreground contains a point
                        for j in range(1, polygon_nuc.max() + 1):
                            nuc = polygon_nuc == j
                            if np.sum(nuc * point_label) > 0:
                                # foreground_instance_img[nuc] = 1
                                instances_embedding.append(pred[nuc > 0])
                    else:
                        # foreground_instance_img[polygon_nuc > 0] = 1
                        instances_embedding.append(pred[polygon_nuc > 0])

        if len(instances_embedding) != 0:
            if bg is not None:
                instances_embedding.append(pred[bg[i] > 0])
            batch_instance_emb.append(instances_embedding)

    return batch_instance_emb


def variance_term(batch_instance_emb, delta_v):
    bz = len(batch_instance_emb)
    var_losses = 0
    batch_mean_cluster = []
    for i in range(bz):
        instances_embedding = batch_instance_emb[i]
        var_loss = 0
        mean_clusters = []
        for emd in instances_embedding:
            mean_cluster = torch.mean(emd, dim=0)
            mean_clusters.append(mean_cluster)

            var_loss += torch.pow(torch.relu(torch.norm(mean_cluster - emd, dim=1) - delta_v), 2).mean()
        var_loss /= len(instances_embedding)
        var_losses += var_loss
        batch_mean_cluster.append(mean_clusters)
    var_losses /= bz

    return var_losses, batch_mean_cluster


def variance_term_bg(batch_instance_emb, delta_v, alpha=None):
    bz = len(batch_instance_emb)
    var_losses = 0
    batch_mean_cluster = []

    for i in range(bz):
        instances_embedding = batch_instance_emb[i][:-1]
        num_fg = len(instances_embedding)
        mean_clusters = []
        bg_emb = batch_instance_emb[i][-1]

        var_loss = 0
        for emd in instances_embedding:
            mean_cluster = torch.mean(emd, dim=0)
            var_loss += torch.pow(torch.relu(torch.norm(mean_cluster - emd, dim=1) - delta_v), 2).mean()
            mean_clusters.append(mean_cluster)

        bg_cluster = torch.mean(bg_emb, dim=0)
        var_loss += alpha * torch.pow(torch.relu(torch.norm(bg_cluster - bg_emb, dim=1) - delta_v), 2).mean()
        mean_clusters.append(bg_cluster)

        var_loss /= (num_fg + 1)
        var_losses += var_loss
        batch_mean_cluster.append(mean_clusters)
    var_losses /= bz

    return var_losses, batch_mean_cluster


def distance_term(batch_mean_cluster, delta_d):
    bz = len(batch_mean_cluster)
    dist_losses = 0
    for i in range(bz):
        mean_clusters = batch_mean_cluster[i]
        n_cluster = torch.arange(len(mean_clusters))
        mean_clusters = torch.stack(mean_clusters, dim=0)
        dist_loss = 0
        if len(n_cluster) <= 1:
            dist_loss = 0
        else:
            for j in n_cluster:
                diff = torch.norm(mean_clusters[j] - mean_clusters[j != n_cluster], dim=1)
                each_cluster_loss = torch.relu(2 * delta_d - diff).pow(2).mean()
                # if torch.isnan(each_cluster_loss):
                #     print("There is NaN")
                #     ipdb.set_trace()
                dist_loss += each_cluster_loss
            dist_loss /= len(n_cluster)

        dist_losses += dist_loss
    dist_losses /= bz

    return dist_losses


def distance_term_bg(batch_mean_cluster, delta_d, alpha=None):
    bz = len(batch_mean_cluster)
    dist_losses = 0
    for i in range(bz):
        mean_clusters = batch_mean_cluster[i][:-1]
        bg_cluster = batch_mean_cluster[i][-1]
        num_fg = len(mean_clusters)

        n_cluster = torch.arange(num_fg)
        mean_clusters = torch.stack(mean_clusters, dim=0)
        dist_loss = 0
        if num_fg <= 1:
            dist_loss = 0
        else:
            for j in n_cluster:
                diff = torch.norm(mean_clusters[j] - mean_clusters[j != n_cluster], dim=1)
                each_cluster_loss = torch.relu(2 * delta_d - diff).pow(2).sum()

                dist_loss += each_cluster_loss

        diff = torch.norm(mean_clusters - bg_cluster, dim=1)
        bg_cluster_loss = 2 * alpha * torch.relu(2 * delta_d - diff).pow(2).sum()
        dist_loss += bg_cluster_loss
        dist_loss /= (num_fg * (num_fg + 1))

        dist_losses += dist_loss
    dist_losses /= bz

    return dist_losses


def regularized_term(batch_mean_cluster):
    bz = len(batch_mean_cluster)
    reg_loss = 0
    for i in range(bz):
        mean_clusters = batch_mean_cluster[i]
        mean_clusters = torch.stack(mean_clusters, dim=0)
        reg_loss += torch.norm(mean_clusters, dim=1).mean()
    reg_loss /= bz
    return reg_loss


class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_v, delta_d, gamma=0.001):
        super(DiscriminativeLoss, self).__init__()
        self.delta_d = delta_d
        self.delta_v = delta_v
        self.gamma = gamma

    def forward(self, embedding, foregrounds, labels, bg=None, alpha=None):

        if labels is None:
            batch_instance_emb = full_ins_emb(embedding, foregrounds, bg=bg)
        else:
            batch_instance_emb = get_instance_emb(embedding, foregrounds, labels, bg=bg)

        if bg is not None:
            var_term, batch_mean_cluster = variance_term_bg(batch_instance_emb, self.delta_v, alpha=alpha)
            dis_term = distance_term_bg(batch_mean_cluster, self.delta_d, alpha=alpha)
        else:
            var_term, batch_mean_cluster = variance_term(batch_instance_emb, self.delta_v)
            dis_term = distance_term(batch_mean_cluster, self.delta_d)
        reg_term = regularized_term(batch_mean_cluster) * self.gamma

        return var_term, dis_term, reg_term


class SoftCrossEntropyWithLogProb(nn.Module):

    def __init__(self, reduction=True):
        super(SoftCrossEntropyWithLogProb, self).__init__()

        self.reduction = reduction

    def forward(self, inputs, soft_label):

        """
        :param inputs: log soft max value
        :param soft_label: soft label (probability) same shape with input
        :return: loss
        """
        raw_loss = -torch.sum(inputs * soft_label, dim=1)

        if self.reduction:
            return torch.mean(raw_loss)
        else:
            return raw_loss


class WeightedNLLLoss2d(nn.Module):
    def __init__(self, ignore_index, weight_pixel=None):
        super(WeightedNLLLoss2d, self).__init__()

        self.weight_pixel = weight_pixel  # shape is C * H * W
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        unweighted_loss = F.nll_loss(inputs, target, weight=None, size_average=None, ignore_index=self.ignore_index,
                                     reduction='none')
        sum_pixels = (target != self.ignore_index).sum()
        if self.weight_pixel is not None:
            indexed_prob_weight = \
                torch.gather(self.weight_pixel, 1, target.unsqueeze(1)).squeeze(1).cuda()

            # get pixel-wise weighted loss (mean reduction)
            sum_loss = torch.sum(indexed_prob_weight * unweighted_loss)
        else:
            sum_loss = torch.sum(unweighted_loss)

        prob_loss = sum_loss / sum_pixels

        return prob_loss
