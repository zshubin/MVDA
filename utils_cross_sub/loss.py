import torch
import torch.nn.functional as F
import torch.nn as nn


class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing=True, eps=0.2):
        super(SmoothCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.eps = eps

    def forward(self, pred, gt):
        gt = gt.contiguous().view(-1)

        if self.smoothing:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, reduction='mean')

        return loss

def discrepancy(out1,out2):
    return torch.mean(torch.abs(out1 - out2))
    # return nn.L1Loss()

def euclidean(x1, x2):
    return ((x1 - x2) ** 2).sum().sqrt()


def k_moment(source_feature, target_feature, k,weight_list):
    source_mean=[]
    for i in range(len(source_feature)):
        source_mean.append((source_feature[i]**k).mean(0))
    target_mean=target_feature.mean(0)
    moment1=0
    for i in range(len(source_feature)):
        for j in range(i+1,len(source_feature)):
            moment1+=euclidean(source_mean[i], source_mean[j])
        moment1+=weight_list[i]*euclidean(source_mean[i],target_mean)
    return moment1


def msda_regulizer(source_feature, target_feature, belta_moment,weight_list):
    source_mean=[]
    for i in range(len(source_feature)):
        source_mean.append(source_feature[i].mean(0))
    target_mean = target_feature.mean(0)
    moment1=0
    for i in range(len(source_feature)):
        for j in range(i+1,len(source_feature)):
            moment1+=euclidean(source_mean[i], source_mean[j])
        moment1+=weight_list[i]*euclidean(source_mean[i],target_mean)
    reg_info = moment1

    for i in range(belta_moment - 1):  # 2->6
        reg_info += k_moment(source_feature, target_feature, i + 2,weight_list)

    return reg_info / 6