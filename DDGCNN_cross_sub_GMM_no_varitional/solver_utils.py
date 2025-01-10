import torch
import numpy as np
import torch.nn as nn
from munkres import Munkres
from torch.autograd import Variable

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot

def filter_samples(samples, threshold=0.05):
    batch_size_full = len(samples['data'])
    min_dist = torch.min(samples['dist2center'], dim=1)[0]
    # sourt_dist = torch.unsqueeze(min_dist,0).sort(1,True)[0]
    # print(sourt_dist)
    mask = min_dist < threshold

    filtered_data = [samples['data'][m]
                     for m in range(mask.size(0)) if mask[m].item() == 1]
    filtered_label = torch.masked_select(samples['label'], mask)
    filtered_gt = torch.masked_select(samples['gt'], mask) \
        if samples['gt'] is not None else None

    filtered_samples = {}
    filtered_samples['data'] = filtered_data
    filtered_samples['label'] = filtered_label
    filtered_samples['gt'] = filtered_gt

    assert len(filtered_samples['data']) == filtered_samples['label'].size(0)
    print('select %f' % (1.0 * len(filtered_data) / batch_size_full))

    return filtered_samples


def filter_class(labels, num_min, num_classes):
    filted_classes = []
    for c in range(num_classes):
        mask = (labels == c)
        count = torch.sum(mask).item()
        if count >= num_min:
            filted_classes.append(c)

    return filted_classes


def split_samples_classwise(samples, num_classes):
    data = samples['data']
    label = samples['label']
    gt = samples['gt']
    samples_list = []
    for c in range(num_classes):
        mask = (label == c)
        data_c = [data[k] for k in range(mask.size(0)) if mask[k].item() == 1]
        label_c = torch.masked_select(label, mask)
        gt_c = torch.masked_select(gt, mask) if gt is not None else None
        samples_c = {}
        samples_c['data'] = data_c
        samples_c['label'] = label_c
        samples_c['gt'] = gt_c
        samples_list.append(samples_c)
    return samples_list


def adjust_learning_rate_exp(lr, optimizer, iters, decay_rate=0.1, decay_step=25):
    lr = lr * (decay_rate ** (iters // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr']


def adjust_learning_rate_RevGrad(lr, optimizer, max_iter, cur_iter, alpha=10, beta=0.75):
    p = 1.0 * cur_iter / (max_iter - 1)
    lr = lr / pow(1.0 + alpha * p, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr']


def adjust_learning_rate_inv(lr, optimizer, iters, alpha=0.001, beta=0.75):
    lr = lr / pow(1.0 + alpha * iters, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr']


def adjust_learning_rate_plateau(optimizer, lr_dacay_rate, optim_patience):
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_dacay_rate,
                                               patience=optim_patience, verbose=True, eps=1e-08)


def set_param_groups(net, lr_mult_dict):
    params = []
    modules = net.module._modules
    for name in modules:
        module = modules[name]
        if name in lr_mult_dict:
            params += [{'params': module.parameters(), 'lr_mult': lr_mult_dict[name]}]
        else:
            params += [{'params': module.parameters(), 'lr_mult': 1.0}]

    return params

def get_centers(net, dataloader, num_classes, device):
    centers = 0
    for index, data in enumerate(dataloader):
        data1, data2, data3, data4, y, path = data[0], data[1], data[2], data[3], data[4], data[5]
        data1, data2, data3, data4, y = data1.to(device), data2.to(device), data3.to(device), data4.to(device), y.to(device)
        batch_size = data1.size(0)
        refs = to_cuda(torch.LongTensor(range(num_classes)).unsqueeze(0))
        refs = refs.expand(batch_size, num_classes)

        mu, sigma, mu_bias, sigma_bias, feature, bias_sample = net(data1, data2, data3, data4)

        y = y.unsqueeze(1).expand(batch_size, num_classes)
        mask = (y == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        feature = feature.unsqueeze(1)
        # update centers
        centers += torch.sum(torch.bmm(mask, feature), dim=0).squeeze()

    return centers

def Sample(z_mean, z_log_var,device):
    std = z_log_var.mul(0.5).exp_()
    eps = Variable(torch.randn(std.size())).to(device)
    return eps.mul(std).add_(z_mean)


def mean_accuracy(preds, target):
    num_classes = preds.size(1)
    preds = torch.max(preds, dim=1).indices
    accu_class = []
    for c in range(num_classes):
        mask = (target == c)
        c_count = torch.sum(mask).item()
        if c_count == 0: continue
        preds_c = torch.masked_select(preds, mask)
        accu_class += [1.0 * torch.sum(preds_c == c).item() / c_count]
    return 100.0 * np.mean(accu_class)

def accuracy(preds, target):
    preds = torch.max(preds, dim=1).indices
    return 100.0 * torch.sum(preds == target).item() / preds.size(0)