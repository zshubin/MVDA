from DDGCNN_cross_sub_MMD_3sub.solver_utils import to_cuda
import torch
import math
import warnings
warnings.filterwarnings("ignore")


class CDD(object):
    def __init__(self, num_layers, kernel_num, kernel_mul,
                 num_classes, intra_only=False, **kwargs):

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.num_classes = num_classes
        self.intra_only = intra_only
        self.num_layers = num_layers

    def split_classwise(self, dist, nums):
        num_classes = len(nums)
        start = end = 0
        dist_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            dist_c = dist[start:end, start:end]
            dist_list += [dist_c]
        return dist_list

    def gamma_estimation(self, dist, flag=None):
        if flag == 'stdis':
            source = list(dist['st'].keys())[0].split('_')[1]
            dist_sum = torch.sum(dist['ss']['s_{}_s_{}'.format(source, source)]) + torch.sum(dist['tt']) + \
                       2 * torch.sum(dist['st']['s_{}_t'.format(source)])

            bs_S = dist['ss']['s_{}_s_{}'.format(source, source)].size(0)
            bs_T = dist['tt'].size(0)
        else:
            dist_sum = 0
            for key in dist['ss'].keys():
                sp_key = key.split('_')
                if sp_key[1]==sp_key[3]:
                    dist_sum += torch.sum(dist['ss'][key])
                else:
                    dist_sum += 2*torch.sum(dist['ss'][key])
                    bs_S = dist['ss']['s_{}_s_{}'.format(sp_key[1],sp_key[1])].size(0)
                    bs_T = dist['ss']['s_{}_s_{}'.format(sp_key[3],sp_key[3])].size(0)

        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / (N+1e-5)

        return gamma

    def patch_gamma_estimation(self, nums_S, nums_T, dist):
        num_classes = len(nums_S[0])
        gammas = {}
        gammas['st'] = {}
        gammas['ss'] = {}
        gammas['tt'] = []
        for key in dist['st'].keys():
            gammas['st'][key] = to_cuda(torch.zeros_like(dist['st'][key], requires_grad=False))
        for key in dist['ss'].keys():
            sp_key = key.split('_')
            if sp_key[1] == sp_key[3]:
                gammas['ss'][key] = []
                for c in range(num_classes):
                    gammas['ss'][key] += [to_cuda(torch.zeros([num_classes], requires_grad=False))]
            else:
                gammas['ss'][key] = to_cuda(torch.zeros_like(dist['ss'][key], requires_grad=False))
        for c in range(num_classes):
            gammas['tt'] += [to_cuda(torch.zeros([num_classes], requires_grad=False))]


        for key in dist['ss'].keys():
            sp_key = key.split('_')
            source1_start = source1_end = 0
            for ns in range(num_classes):
                source1_start = source1_end
                source1_end = source1_start + nums_S[int(sp_key[1])][ns]
                patch = {}
                patch['ss'] = {}
                patch['st'] = {}
                if sp_key[1] == sp_key[3]:
                    patch['ss'][key] = dist['ss'][key][ns]
                else:
                    patch['ss']['s_{}_s_{}'.format(sp_key[1], sp_key[1])] = dist['ss']['s_{}_s_{}'.format(sp_key[1], sp_key[1])][ns]
                if sp_key[1] == sp_key[3]:
                    target_start = target_end = 0
                    for nt in range(num_classes):
                        target_start = target_end
                        target_end = target_start + nums_T[nt]
                        patch['tt'] = dist['tt'][nt]
                        patch['st']['s_{}_t'.format(sp_key[1])] = dist['st']['s_{}_t'.format(sp_key[1])]
                        gamma = self.gamma_estimation(patch, 'stdis')
                        gammas['ss']['s_{}_s_{}'.format(sp_key[1],sp_key[1])][ns][nt] += gamma
                        gammas['tt'][nt][ns] += gamma
                        gammas['st']['s_{}_t'.format(sp_key[1])][source1_start:source1_end, target_start:target_end] = gamma
                else:
                    source2_start = source2_end = 0
                    for ns_2 in range(num_classes):
                        source2_start = source2_end
                        source2_end = source2_start + nums_S[int(sp_key[3])][ns_2]
                        patch['ss'][key] = dist['ss'][key].narrow(0, source1_start,nums_S[int(sp_key[1])][ns]).narrow(1, source2_start, nums_S[int(sp_key[3])][ns_2])
                        patch['ss']['s_{}_s_{}'.format(sp_key[3], sp_key[3])] = dist['ss']['s_{}_s_{}'.format(sp_key[3], sp_key[3])][ns_2]
                        gamma = self.gamma_estimation(patch, 's_dis')
                        gammas['ss']['s_{}_s_{}'.format(sp_key[1], sp_key[1])][ns][ns_2] += gamma
                        gammas['ss']['s_{}_s_{}'.format(sp_key[3], sp_key[3])][ns_2][ns] += gamma
                        gammas['ss'][key][source1_start:source1_end, source2_start:source2_end] = gamma
        for key in gammas['ss'].keys():
            sp_key = key.split('_')
            if sp_key[1]==sp_key[3]:
                for i in range(len(gammas['ss'][key])):
                    gammas['ss'][key][i] /= float(len(nums_S))
        for i in range(len(gammas['tt'])):
            gammas['tt'][i] /= float(len(nums_S))

        return gammas

    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul ** i) for i in range(kernel_num)]
        gamma_tensor = to_cuda(torch.stack(gamma_list, dim=0))

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps
        gamma_tensor = gamma_tensor.detach()

        for i in range(len(gamma_tensor.size()) - len(dist.size())):
            dist = dist.unsqueeze(0)

        dist = dist / gamma_tensor
        upper_mask = (dist > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dist < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers, kernel_num, kernel_mul, category=None):
        dist = dist_layers if category is None else dist_layers[category]
        gamma = gamma_layers if category is None else gamma_layers[category]
        kernel_dist = self.compute_kernel_dist(dist, gamma, kernel_num, kernel_mul)
        return kernel_dist

    def patch_mean(self, nums_row, nums_col, dist):
        assert (len(nums_row) == len(nums_col))
        num_classes = len(nums_row)

        mean_tensor = to_cuda(torch.zeros([num_classes, num_classes]))
        row_start = row_end = 0
        for row in range(num_classes):
            row_start = row_end
            row_end = row_start + nums_row[row]

            col_start = col_end = 0
            for col in range(num_classes):
                col_start = col_end
                col_end = col_start + nums_col[col]
                val = torch.mean(dist.narrow(0, row_start, nums_row[row]).narrow(1, col_start, nums_col[col]))
                mean_tensor[row, col] = val
        return mean_tensor

    def compute_paired_dist(self, A, B):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dist = (((A_expand - B_expand)) ** 2).sum(2)
        return dist

    def forward(self, source, target, nums_S, nums_T):

        num_classes = len(nums_S[0])

        # compute the dist
        dist_layers = []
        gamma_layers = []

        for i in range(self.num_layers):

            dist = {}
            dist['ss'] = {}
            dist['st'] = {}
            for m in range(len(source)):
                for n in range(m, len(source)):
                    dist['ss']['s_{}_s_{}'.format(m,n)] = self.compute_paired_dist(source[m], source[n])
                    if m == n:
                       dist['ss']['s_{}_s_{}'.format(m,n)] = self.split_classwise(dist['ss']['s_{}_s_{}'.format(m,n)], nums_S[m])

                dist['st']['s_{}_t'.format(m)] = self.compute_paired_dist(source[m], target)

            dist['tt'] = self.compute_paired_dist(target, target)
            dist['tt'] = self.split_classwise(dist['tt'], nums_T)

            dist_layers += [dist]

            gamma_layers += [self.patch_gamma_estimation(nums_S, nums_T, dist)]

        # compute the kernel dist   `
        for i in range(self.num_layers):
            for c in range(num_classes):
                for key in gamma_layers[i]['ss'].keys():
                    sp_key = key.split('_')
                    if sp_key[1]==sp_key[3]:
                       gamma_layers[i]['ss'][key][c] = gamma_layers[i]['ss'][key][c].view(num_classes, 1, 1)
                gamma_layers[i]['tt'][c] = gamma_layers[i]['tt'][c].view(num_classes, 1, 1)

        kernel_dist_st = {}
        for i in range(self.num_layers):
            for key in dist_layers[i]['st'].keys():
                if key not in kernel_dist_st:
                    kernel_dist_st[key] = self.kernel_layer_aggregation(dist_layers[i]['st'][key], gamma_layers[i]['st'][key], self.kernel_num[i], self.kernel_mul[i])
                else:
                    kernel_dist_st[key] += self.kernel_layer_aggregation(dist_layers[i]['st'][key], gamma_layers[i]['st'][key], self.kernel_num[i], self.kernel_mul[i])
        for key in kernel_dist_st.keys():
            sp_key = key.split('_')
            kernel_dist_st[key] = self.patch_mean(nums_S[int(sp_key[1])], nums_T, kernel_dist_st[key])

        kernel_dist_ss = {}
        for i in range(self.num_layers):
            for key in dist_layers[i]['ss'].keys():
                sp_key = key.split('_')
                if sp_key[1] != sp_key[3]:
                    if key not in kernel_dist_ss:
                        kernel_dist_ss[key] = self.kernel_layer_aggregation(dist_layers[i]['ss'][key], gamma_layers[i]['ss'][key], self.kernel_num[i], self.kernel_mul[i])
                    else:
                        kernel_dist_ss[key] += self.kernel_layer_aggregation(dist_layers[i]['ss'][key], gamma_layers[i]['ss'][key], self.kernel_num[i], self.kernel_mul[i])

        for key in kernel_dist_ss.keys():
            sp_key = key.split('_')
            kernel_dist_ss[key] = self.patch_mean(nums_S[int(sp_key[1])], nums_S[int(sp_key[3])], kernel_dist_ss[key])

        kernel_dist_tt = []
        for c in range(num_classes):
            for i in range(self.num_layers):
                for key in dist_layers[i]['ss'].keys():
                    sp_key = key.split('_')
                    if sp_key[1] == sp_key[3]:
                        if key not in kernel_dist_ss:
                            kernel_dist_ss[key] = []
                            kernel_dist_ss[key] += [(torch.mean(self.kernel_layer_aggregation(dist_layers[i]['ss'][key], gamma_layers[i]['ss'][key],
                                                                                        self.kernel_num[i], self.kernel_mul[i], c).view(num_classes, -1), dim=1))]
                        else:
                            kernel_dist_ss[key][-1] += torch.mean(self.kernel_layer_aggregation(dist_layers[i]['ss'][key], gamma_layers[i]['ss'][key],
                                                                                        self.kernel_num[i], self.kernel_mul[i], c).view(num_classes, -1), dim=1)

                kernel_dist_tt += [torch.mean(self.kernel_layer_aggregation(dist_layers[i]['tt'], gamma_layers[i]['tt'], self.kernel_num[i], self.kernel_mul[i], c).view(num_classes, -1),dim=1)]
        for key in kernel_dist_ss.keys():
            sp_key = key.split('_')
            if sp_key[1] == sp_key[3]:
              kernel_dist_ss[key] = torch.stack(kernel_dist_ss[key], dim=0)
        kernel_dist_tt = torch.stack(kernel_dist_tt, dim=0).transpose(1, 0)

        mmds = 0.
        for key in kernel_dist_ss.keys():
            sp_key = key.split('_')
            if sp_key[1] == sp_key[3]:
                mmds += len(nums_S)*kernel_dist_ss[key]
            else:
                mmds -= 2*kernel_dist_ss[key]
        for key in kernel_dist_st.keys():
            mmds -= 2*kernel_dist_st[key]
        mmds += len(nums_S)*kernel_dist_tt
        # mmds = kernel_dist_ss + kernel_dist_tt - 2 * kernel_dist_st
        intra_mmds = torch.diag(mmds, 0)
        intra = torch.sum(intra_mmds) / self.num_classes
        inter = None
        if not self.intra_only:
            inter_mask = to_cuda((torch.ones([num_classes, num_classes]) - torch.eye(num_classes)).type(torch.bool))
            inter_mmds = torch.masked_select(mmds, inter_mask)
            inter = torch.sum(inter_mmds) / (self.num_classes * (self.num_classes - 1))

        cdd = intra if inter is None else intra - inter

        return {'cdd': cdd, 'intra': intra, 'inter': inter}
