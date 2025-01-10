# from DDGCNN_cross_sub_GMM_no_varitional.cdd_model import Extractor, Predictor
# from DDGCNN_cross_sub_GMM_no_varitional.config import OptInit
# import logging
# from data_loader_cross_sub.prepare_data import *
# from DDGCNN_cross_sub_GMM_no_varitional.subject_measure import subject_sim_measure
# import sys
# from DDGCNN_cross_sub_GMM_no_varitional.can_solver import CANSolver as Solver
#
#
# def train(logger, sess, source_sub, tar_sub, rest_sub):
#     opt_class = OptInit(tar_sub, logger)
#     opt = opt_class.get_args()
#     data_path = []
#     for i in source_sub:
#         data_path.append(os.path.join('./processed_ssvep_data/', sess, i))
#     data_path.append(os.path.join('./processed_ssvep_data/', sess, tar_sub))
#     rest_sub_path = []
#     for i in rest_sub:
#         rest_sub_path.append(os.path.join('./processed_ssvep_data/', sess, i))
#     dataloaders = prepare_data_CAN(opt.batch_size,data_path,rest_sub_path,opt.time_win,opt.down_sample,opt.sample_freq,opt.device)
#
#     Extractor_model = Extractor([opt.batch_size, opt.in_channels, opt.eeg_channel], opt.k_adj, opt.n_filters, opt.dropout,
#                       bias=opt.bias, norm=opt.norm, act=opt.act, trans_class=opt.trans_class, device=opt.device).to(opt.device)
#     # extractor_state_dict = torch.load('./DDGCNN_cross_sub_GMM_4sub/save_model/s43/model.pth', map_location=torch.device('cpu'))['extractor_state_dict']
#     extractor_state_dict = torch.load('./DDGCNN_cross_sub_GMM_no_varitional/save_model/s43/model.pth', map_location=torch.device('cpu'))['extractor_state_dict']
#     Extractor_model.load_state_dict(extractor_state_dict)
#     Extractor_model.eval()
#     target_data_loader = dataloaders['target_test']
#     feature_list = []
#     label_list = []
#     for i, tar_batch in enumerate(target_data_loader):
#         data1, data2, data3, data4, y, path = tar_batch[0], tar_batch[1], tar_batch[2], tar_batch[3], tar_batch[4], tar_batch[5]
#         tar_feature = Extractor_model(data1, data2, data3, data4)
#         tar_feature = tar_feature.detach().numpy()
#         feature_list.append(np.expand_dims(tar_feature,axis=0))
#         label_list.append(np.expand_dims(y.numpy(),axis=0))
#     feature = np.concatenate(feature_list,axis=0)
#     label = np.concatenate(label_list, axis=0)
#     np.save('./no_varitional_T_S43_S_S48_feature', feature)
#     np.save('./no_varitional_T_S43_S_S48_label', label)
#     print(1)
#
#
#
# def make_logger():
#     loglevel = "info"
#     numeric_level = getattr(logging, loglevel.upper(), None)
#     if not isinstance(numeric_level, int):
#         raise ValueError('Invalid log level: {}'.format(loglevel))
#     log_format = logging.Formatter('%(asctime)s %(message)s')
#     logger = logging.getLogger()
#     logger.setLevel(numeric_level)
#     file_handler = logging.StreamHandler(sys.stdout)
#     file_handler.setFormatter(log_format)
#     logger.addHandler(file_handler)
#     logging.root = logger
#     return logger
#
#
# if __name__ == '__main__':
#     logger = make_logger()
#     source_num = 4
#     sess = 'session1'
#     target_subject_list = ['s48']
#     for i in range(len(target_subject_list)):
#         target_subject = target_subject_list[i]
#         source_subject_list =['s1']
#         rest_subject_list = ['s1']
#         train(logger, sess, source_subject_list, target_subject, rest_subject_list)


import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
import math
import mpl_toolkits.mplot3d

def visual(feat):
    ts = manifold.TSNE(n_components=2, init='pca', random_state=2022,learning_rate=180)#200,180
    x_ts = ts.fit_transform(feat)
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final

maker = ['s', '+', 'o', '*', 'v']
colors = ['darkorange', 'blue', 'red', 'darksalmon','cornflowerblue']
Label_Com = ['a', 'b', 'c', 'd','e']
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# plt.figure(figsize=(10,7))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=15)

def plotlabels(S_lowDWeights, True_labels):
    # True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    for index in range(5):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=10, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.5)
        plt.xticks([])
        plt.yticks([])
feat1 = np.load('./no_varitional_T_S43_feature.npy')
label1 = np.ones((feat1.shape[0],1))*0
# label1 = np.load('./varitional_T_S43_label.npy')

feat2 = np.load('./no_varitional_T_S43_S_S14_feature.npy')
label2 = np.ones((feat1.shape[0],1))*1
# label2 = np.load('./varitional_T_S43_S_S14_label.npy')

feat3 = np.load('./no_varitional_T_S43_S_S37_feature.npy')
label3 = np.ones((feat1.shape[0],1))*2
# label3 = np.load('./varitional_T_S43_S_S37_label.npy')

feat4 = np.load('./no_varitional_T_S43_S_S38_feature.npy')
label4 = np.ones((feat1.shape[0],1))*3
# label4 = np.load('./varitional_T_S43_S_S38_label.npy')

feat5 = np.load('./no_varitional_T_S43_S_S48_feature.npy')
label5 = np.ones((feat1.shape[0],1))*4
# label5 = np.load('./varitional_T_S43_S_S48_label.npy')

feat = np.concatenate([feat1,feat2,feat3,feat4,feat5],axis=0)
label = np.concatenate([label1,label2,label3,label4,label5],axis=0)
tsne_feat = visual(feat)
plotlabels(tsne_feat,label)
plt.legend(['S43','S14','S37','S38','S48'])
# plt.legend(['Class 1','Class 2','Class 3','Class 4'])
# plt.show()

plt.savefig("./no_varitional_diff_source.pdf",dpi=800)
