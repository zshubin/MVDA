import torch
import torch.nn as nn
import torch.nn.functional as F
from DDGCNN_cross_sub_MKMMD.layers import GraphConvolution,Linear,GDCD,grad_reverse
from DDGCNN_cross_sub_MKMMD.utils import normalize_A, generate_cheby_adj, randomedge_drop
from DDGCNN_cross_sub_MKMMD.solver_utils import Sample


class DCDGCN(nn.Module):
    def __init__(self, num_nodes, xdim, K, num_out, dropout=0., bias=True, norm='batch', act='relu',trans_class='DCD', device='cuda'):
        super(DCDGCN, self).__init__()
        self.device = device
        self.K = K
        self.gc = nn.ModuleList()
        self.norm, self.act = None, None
        self.A = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes).to(self.device))
        nn.init.xavier_normal_(self.A)
        self.dropout = dropout
        if dropout != 0.:
            self.droplayer = nn.Dropout(dropout)
        else:
            self.droplayer = False
        for i in range(K):
            self.gc.append(GraphConvolution(xdim, num_out, bias=bias, dropout=self.dropout, trans_class=trans_class,device=self.device))
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(num_out)
        if norm == 'layer':
            self.norm = nn.LayerNorm([num_out, num_nodes, 1], elementwise_affine=True)
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(num_out, affine=True)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'Leakyrelu':
            self.act = nn.LeakyReLU(0.2, inplace=False).to(self.device)
        else:
            self.act = nn.PReLU(1, 0.2).to(self.device)

    def forward(self, x):
        L = normalize_A(self.A, self.device)
        adj = generate_cheby_adj(L, self.K, self.device)
        for i in range(self.K):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        if self.norm:
            self.norm = self.norm.to(self.device)
            result = self.norm(result)
        if self.act:
            result = self.act(result)
        if self.droplayer:
            result = self.droplayer(result)
        return result


class Extractor(nn.Module):
    def __init__(self, xdim, k_adj, num_out, dropout, bias=True, norm='batch', act='relu', trans_class='DCD', device='cuda'):
        super(Extractor, self).__init__()
        self.K = k_adj
        self.num_out = num_out
        self.bias = bias
        self.device = device
        self.dropout = dropout
        self.num_nodes = xdim[2]
        if self.dropout is not None and self.dropout > 0:
            self.droplayer1 = nn.Dropout(self.dropout)
            self.droplayer2 = nn.Dropout(self.dropout)
            self.droplayer_bias1 = nn.Dropout(self.dropout)
            self.droplayer_bias2 = nn.Dropout(self.dropout)
            self.droplayer_bias3 = nn.Dropout(self.dropout)
            self.droplayer_bias4 = nn.Dropout(self.dropout)

        if act == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
            self.act4 = nn.ReLU()
            self.act_bias3 = nn.ReLU()
            self.act_bias4 = nn.ReLU()
        elif act == 'Leakyrelu':
            self.act1 = nn.LeakyReLU(0.2, inplace=False).to(self.device)
            self.act2 = nn.LeakyReLU(0.2, inplace=False).to(self.device)
            self.act3 = nn.LeakyReLU(0.2, inplace=False).to(self.device)
            self.act4 = nn.LeakyReLU(0.2, inplace=False).to(self.device)
            self.act_bias3 = nn.LeakyReLU(0.2, inplace=False).to(self.device)
            self.act_bias4 = nn.LeakyReLU(0.2, inplace=False).to(self.device)
        else:
            self.act1 = nn.PReLU(1, 0.2).to(self.device)
            self.act2 = nn.PReLU(1, 0.2).to(self.device)
            self.act3 = nn.PReLU(1, 0.2).to(self.device)
            self.act4 = nn.PReLU(1, 0.2).to(self.device)
            self.act_bias3 = nn.PReLU(1, 0.2).to(self.device)
            self.act_bias4 = nn.PReLU(1, 0.2).to(self.device)

        self.norm1 = nn.BatchNorm2d(num_out*2)
        self.norm2 = nn.BatchNorm2d(num_out*4)

        self.norm3 = nn.BatchNorm2d(num_out*8)
        self.norm4 = nn.BatchNorm2d(num_out*8)
        self.norm5 = nn.BatchNorm2d(num_out*8)
        self.norm6 = nn.BatchNorm2d(num_out*8)

        self.trans_class = trans_class
        self.device = device
        self.num_features = xdim[1]
        self.bottle_neck1 = DCDGCN(self.num_nodes, self.num_features, self.K, num_out, dropout=0.,  #62 50
                               bias=bias, norm=norm, act=act,trans_class=trans_class, device=device)
        self.bottle_neck2 = DCDGCN(self.num_nodes, self.num_features, self.K, num_out, dropout=0.,
                               bias=bias, norm=norm, act=act,trans_class=trans_class, device=device)
        self.bottle_neck3 = DCDGCN(self.num_nodes, self.num_features, self.K, num_out, dropout=0.,
                               bias=bias, norm=norm, act=act,trans_class=trans_class, device=device)
        self.bottle_neck4 = DCDGCN(self.num_nodes, self.num_features, self.K, num_out, dropout=0.,
                               bias=bias, norm=norm, act=act,trans_class=trans_class, device=device)
        self.conv1 = nn.Conv2d(num_out, num_out*2, kernel_size=(62, 1), stride=(1, 1), bias=bias)  #187
        self.conv2 = nn.Conv2d(num_out*2, num_out*4, kernel_size=(62, 1), stride=(1, 1), bias=bias)  #126

        self.conv3 = nn.Conv2d(num_out*4, num_out*8, kernel_size=(62, 1), stride=(1, 1), bias=bias)  #65
        self.conv4 = nn.Conv2d(num_out*4, num_out*8, kernel_size=(62, 1), stride=(1, 1), bias=bias)  #65

        self.conv_bias3 = nn.Conv2d(num_out*4, num_out*8, kernel_size=(62, 1), stride=(1, 1), bias=bias)  #65
        self.conv_bias4 = nn.Conv2d(num_out*4, num_out*8, kernel_size=(62, 1), stride=(1, 1), bias=bias)  #65

    def forward(self, x1, x2, x3, x4):
        result1 = self.bottle_neck1(x1)
        result2 = self.bottle_neck2(x2)
        result3 = self.bottle_neck3(x3)
        result4 = self.bottle_neck4(x4)
        result_concat = torch.cat((result1, result2, result3, result4), dim=2)

        result = self.conv1(result_concat)
        result = self.norm1(result).to(self.device)
        result = self.act1(result)
        if self.dropout:
            result = self.droplayer1(result)

        result = self.conv2(result)
        result = self.norm2(result).to(self.device)
        result = self.act2(result)
        if self.dropout:
            result = self.droplayer2(result)


        sigma = self.conv4(result)
        sigma = self.norm3(sigma).to(self.device)
        sigma = sigma.flatten(1,3)
        sigma = torch.squeeze(sigma)

        return sigma

    def get_last_shared_layer(self):
        return self.conv2


class Predictor(nn.Module):
    def __init__(self, kernel_num, n_blocks=0, nclass=4):
        super(Predictor, self).__init__()
        self.fc = Linear(65*kernel_num*8*n_blocks, nclass)

    def forward(self, sample):
        result = self.fc(sample)
        return result


