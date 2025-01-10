import torch
from torch.nn import functional as F
from DDGCNN_cross_sub_GMM_3sub.solver_utils import to_cuda, to_onehot
from scipy.optimize import linear_sum_assignment
from math import ceil
import numpy as np


class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=True):
        return getattr(self, self.dist_type)(
            pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert (pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))


class Clustering(object):
    def __init__(self, eps, max_len=1000, dist_type='cos'):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = 0
        self.stop = False
        self.max_len = max_len

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.source_center = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers[0].size(0)

    def clustering_stop(self, centers):
        if len(centers) == 0:
            self.stop = False
        else:
            dist = 0.
            for i in range(len(centers)):
                d = self.Dist.get_dist(centers[i], self.centers[i],cross=False)
                dist += torch.mean(d, dim=0)
            dist/= len(centers)
            self.stop = dist.item() < self.eps or self.cluster_iter>1000

    def assign_labels(self, feats):
        dists = []
        labels = []
        for i in range(len(self.centers)):
            dists.append(self.Dist.get_dist(feats, self.centers[i], cross=True))
            _, label = torch.min(dists[i], dim=1)
            labels.append(label)
        return dists, labels

    def assign_labels_AE(self, feats):
        pr_c = 0
        dists = 0
        for i in range(len(self.centers)):
            dists += self.Dist.get_dist(feats,self.centers[i],cross=True)
            miu_e = torch.mean(self.Dist.get_dist(feats, self.centers[i], cross=True), dim=1)
            sigma_e = torch.sqrt(torch.sum(torch.square(self.Dist.get_dist(feats, self.centers[i],cross=True) - miu_e.unsqueeze(1).expand(feats.shape[0], self.num_classes)),dim=1)
                                 / (self.num_classes - 1))
            pr_c += torch.exp(-(self.Dist.get_dist(feats, self.centers[i], cross=True) - miu_e.unsqueeze(1).expand(feats.shape[0],self.num_classes)) / sigma_e.unsqueeze(1).expand(feats.shape[0], self.num_classes)) / torch.sum(torch.exp(-(
                        self.Dist.get_dist(feats, self.centers[i], cross=True) - miu_e.unsqueeze(1).expand(feats.shape[0],self.num_classes)) / sigma_e.unsqueeze(1).expand(feats.shape[0], self.num_classes)),
                        dim=1).unsqueeze(1).expand(feats.shape[0], self.num_classes) # class conditional prob of each centers  (shape-> sample_size,class)

        dists/=len(self.centers)
        _, label = torch.max(pr_c, dim=1)   #
        return dists, label

    def assign_labels_IWE(self, feats, Iwe_list):
        pr_c = 0
        dists = 0
        for i in range(len(self.centers)):
            dists += Iwe_list[i]*self.Dist.get_dist(feats,self.centers[i],cross=True)
            miu_e = torch.mean(self.Dist.get_dist(feats, self.centers[i], cross=True), dim=1)
            sigma_e = torch.sqrt(torch.sum(torch.square(self.Dist.get_dist(feats, self.centers[i],cross=True) - miu_e.unsqueeze(1).expand(feats.shape[0], self.num_classes)),dim=1) / (self.num_classes - 1))
            pr_c += Iwe_list[i]*torch.exp(-(self.Dist.get_dist(feats, self.centers[i], cross=True) - miu_e.unsqueeze(1).expand(feats.shape[0],self.num_classes)) / sigma_e.unsqueeze(1).expand(feats.shape[0], self.num_classes)) / torch.sum(torch.exp(-(
                        self.Dist.get_dist(feats, self.centers[i], cross=True) - miu_e.unsqueeze(1).expand(feats.shape[0],self.num_classes)) / sigma_e.unsqueeze(1).expand(feats.shape[0], self.num_classes)),
                        dim=1).unsqueeze(1).expand(feats.shape[0], self.num_classes)
        _, label = torch.max(pr_c, dim=1)
        return dists, label

    def align_centers(self,Iwe_list):
        cost = 0
        for i in range(len(self.centers)):
            cost += self.Dist.get_dist((Iwe_list[i]*self.centers[i].transpose(0,1)).transpose(0,1),(Iwe_list[i]*self.init_centers[i].transpose(0,1)).transpose(0,1), cross=True)
        cost = cost.data.cpu().numpy()
        cost[np.isnan(cost)] = 1e+2
        cost[np.isinf(cost)] = 0
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def collect_samples(self, net, loader,device):
        data_feat, data_gt, data_paths = [], [], []
        for index, data in enumerate(loader):
            data1, data2, data3, data4, y , path= data[0], data[1], data[2], data[3], data[4], data[5]
            data1, data2, data3, data4, y = data1.to(device), data2.to(device), data3.to(device), data4.to(
                device), y.to(device)
            data_paths += path
            data_gt += [y]
            mu, sigma, mu_bias, sigma_bias, sample, feature = net(data1, data2, data3, data4)
            data_feat += [feature]

        self.samples['data'] = data_paths
        self.samples['gt'] = torch.cat(data_gt, dim=0) \
            if len(data_gt) > 0 else None
        self.samples['feature'] = torch.cat(data_feat, dim=0)

    def feature_clustering(self, net, loader,device):
        centers_list = []
        self.stop = False

        self.collect_samples(net, loader,device=device)
        feature = self.samples['feature']

        refs = to_cuda(torch.LongTensor(range(self.num_classes)).unsqueeze(1))
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)
        self.cluster_iter = 0
        while True:
            self.clustering_stop(centers_list)
            if len(centers_list) != 0:
                self.centers = centers_list
            if self.stop: break
            self.cluster_iter += 1

            ###########IWE multi-source##############
            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                ensamble_prob, labels = self.assign_labels_AE(cur_feature)
                labels_onehot = to_onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len
            sum_I_e = 0
            I_e_list = []
            for ii in range(len(self.source_center)):
                I_e = 1-torch.mean(self.Dist.get_dist(centers,self.source_center[ii],cross=True), dim=1)
                I_e_list.append(I_e)
                sum_I_e += I_e
            miu_e = sum_I_e/len(I_e_list)
            sigma_e = 0.
            for ii in range(len(self.source_center)):
                sigma_e += torch.square(I_e_list[ii]-miu_e)
            sigma_e = torch.sqrt(sigma_e/(len(self.source_center)-1))
            Iwe_list = []
            Iwe_sum = 0
            for ii in range(len(self.source_center)):
                Iwe = torch.exp((I_e_list[ii]-miu_e)/sigma_e)
                Iwe_list.append(Iwe)
                Iwe_sum += Iwe
            for ii in range(len(self.source_center)):
                Iwe_list[ii] /= Iwe_sum

            centers = 0
            count = 0
            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                ensamble_prob, labels = self.assign_labels_IWE(cur_feature,Iwe_list)
                labels_onehot = to_onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len

            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)
            centers_list = []
            for ii in range(len(self.init_centers)):
                centers_list.append(mask * centers + (1 - mask) * self.init_centers[ii])

            ###################################################

        dist2center, labels = [], []
        start = 0
        count = []
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels_IWE(cur_feature,Iwe_list)
            labels_onehot = to_onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers(Iwe_list)
        # reorder the centers
        for i in range(len(self.centers)):
          self.centers[i] = self.centers[i][cluster2label, :]
        # re-label the data according to the index
        num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()

        for i in range(len(self.centers)):
            self.center_change += torch.mean(self.Dist.get_dist(self.centers[i], self.init_centers[i]))
        self.center_change /= len(self.centers)

        for i in range(num_samples):
            self.path2label[self.samples['data'][i]] = self.samples['label'][i].item()

        del self.samples['feature']

def compute_paired_dist(A, B):
    bs_A = A.size(0)
    bs_T = B.size(0)
    feat_len = A.size(1)

    A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
    B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
    dist = (((A_expand - B_expand)) ** 2).sum(2)
    return dist

if __name__ =="__main__":
    dist = DIST(dist_type='cos')
    feat = torch.rand(1,256)
    centers = torch.rand(4,256)
    miu_e = torch.mean(dist.get_dist(feat, centers, cross=True), dim=0)
    sigma_e = torch.sqrt(torch.sum(torch.square(dist.get_dist(feat, centers) - miu_e.repeat(feat.shape[0], feat.shape[0])),
                  dim=0) / (4 - 1))
    p_r_c = torch.exp( -(dist.get_dist(feat, centers, cross=True) - miu_e.repeat(feat.shape[0], feat.shape[0])) / sigma_e) \
            / torch.sum(torch.exp(-(dist.get_dist(feat, centers, cross=True) - miu_e.repeat(feat.shape[0], feat.shape[0])) / sigma_e),dim=0)
    print(p_r_c.shape)
    source_list1 = [[torch.rand(10,64,100),torch.rand(10,64,100),torch.rand(10,64,100)],
                   [torch.rand(10, 64, 100), torch.rand(10, 64, 100), torch.rand(10, 64, 100)]]
    source_list2 = [[torch.rand(10,64,100),torch.rand(10,64,100),torch.rand(10,64,100)],
                   [torch.rand(10, 64, 100), torch.rand(10, 64, 100), torch.rand(10, 64, 100)]]
    source_list = [source_list1,source_list2]
    target_list = [[torch.rand(10,64,100),torch.rand(10,64,100),torch.rand(10,64,100)],
                   [torch.rand(10, 64, 100), torch.rand(10, 64, 100), torch.rand(10, 64, 100)]]
    for j, (cdd_s, cdd_tar) in enumerate(zip(zip(source_list1,source_list2), target_list)):
        for index, data in enumerate(cdd_s):
            if j == 0:
                source_cls_concat1 = data[0]
                source_cls_concat2 = data[1]
                source_cls_concat3 = data[2]
            else:
                cource_cls_concat1 = torch.cat([source_cls_concat1, data[0]], dim=0)
                cource_cls_concat2 = torch.cat([source_cls_concat2, data[1]], dim=0)
                cource_cls_concat3 = torch.cat([source_cls_concat3, data[2]], dim=0)

        if j == 0:
            target_cls_concat1 = cdd_tar[0]
            target_cls_concat2 = cdd_tar[1]
            target_cls_concat3 = cdd_tar[2]
        else:
            target_cls_concat1 = torch.cat([target_cls_concat1, cdd_tar[0]], dim=0)
            target_cls_concat2 = torch.cat([target_cls_concat2, cdd_tar[1]], dim=0)
            target_cls_concat3 = torch.cat([target_cls_concat2, cdd_tar[2]], dim=0)
    feat_a = torch.rand(10,256)
    feat_b = torch.rand(20,256)
    dist = compute_paired_dist(feat_a,feat_b)
    print(dist)
print(1)




