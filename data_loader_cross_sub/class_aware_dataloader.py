import torch.utils.data
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import scipy.signal as sp
from scipy.signal import butter, lfilter
import os

def collate_fn(data):
    # data is a list: index indicates classes
    data_collate = {}
    num_classes = len(data)
    keys = data[0].keys()
    for key in keys:
        if key.find('Label') != -1:
            data_collate[key] = [torch.tensor(data[i][key]) for i in range(num_classes)]
        if key.find('Img') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]
        if key.find('Path') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]

    return data_collate


def butter_bandpass_filter(data, lowcut=8, highcut=30, samplingRate=512, order=4):
    y = np.zeros_like(data).astype(np.float32)
    nyq = 0.5 * samplingRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    for i in range(data.shape[0]):
        y[i, :] = lfilter(b, a, data[i, :])
    return y


def iir_bandpass_filter(data, lowcut=8, highcut=30, samplingRate=512, order=4):
    y = np.zeros_like(data).astype(np.float32)
    nyq = 0.5 * samplingRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp.iirfilter(order, [low, high], btype='band')
    for i in range(data.shape[0]):
        y[i, :] = sp.filtfilt(b, a, data[i, :],padlen=0)
    return y


class DatasetProcessing(Dataset):
    def __init__(self, target_path, source_path, target_gt, win_train=1, down_sample=1, sample_freq=1000, transform=None, device='cuda'):
        self.transform = transform
        self.bci_file_name = []
        self.targt_gt = target_gt
        self.sample_freq = int(sample_freq/down_sample)
        self.win_train = int(self.sample_freq*win_train)
        self.source_path = source_path
        self.down_sample = down_sample
        self.class_list = list(target_path.keys())
        self.source_list = []
        for p in self.source_path:
            self.source_list.append([os.listdir(os.path.join(p, 'train', str(j+1))) for j in range(4)])
        self.label = []
        self.device = device
        self.class_dict = {}
        self.target_path = target_path
        self.min_target_length = min(len(self.target_path[i]) for i in list(self.target_path.keys()))
        self.min_target_length = min(self.min_target_length, 22)
        self.iter = self.min_target_length // 4
        if self.min_target_length % 4 != 0:
            self.whole_iter = self.iter + 1
        else:
            self.whole_iter = self.iter
        self.batch_size = 4

    def target_data_reader(self, class_index, sample_index):
        def simple_batch_norm_1d(x,dim):
            eps = 1e-5
            x_mean = torch.mean(x, dim=dim, keepdim=True)
            x_var = torch.mean((x - x_mean) ** 2, dim=dim, keepdim=True)
            x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
            return x_hat
        time_start = random.randint(35, int(self.sample_freq*4 + 35 - self.win_train))
        x1 = time_start
        x2 = time_start + self.win_train
        bci_data = np.load(self.target_path[class_index][sample_index])[:, ::self.down_sample][:, x1:x2]
        bci_data1 = iir_bandpass_filter(bci_data, 3, 14, self.sample_freq, 4).astype(np.float32)
        bci_data2 = iir_bandpass_filter(bci_data, 9, 26, self.sample_freq, 4).astype(np.float32)
        bci_data3 = iir_bandpass_filter(bci_data, 14, 38, self.sample_freq, 4).astype(np.float32)
        bci_data4 = iir_bandpass_filter(bci_data, 19, 50, self.sample_freq, 4).astype(np.float32)

        bci_data1 = torch.from_numpy(bci_data1.T).to(self.device)
        bci_data2 = torch.from_numpy(bci_data2.T).to(self.device)
        bci_data3 = torch.from_numpy(bci_data3.T).to(self.device)
        bci_data4 = torch.from_numpy(bci_data4.T).to(self.device)
        if self.transform is None:
            bci_data1 = simple_batch_norm_1d(bci_data1,dim=1)
            bci_data2 = simple_batch_norm_1d(bci_data2,dim=1)
            bci_data3 = simple_batch_norm_1d(bci_data3,dim=1)
            bci_data4 = simple_batch_norm_1d(bci_data4,dim=1)
        bci_data1 = torch.unsqueeze(torch.unsqueeze(bci_data1, 2),0)
        bci_data2 = torch.unsqueeze(torch.unsqueeze(bci_data2, 2),0)
        bci_data3 = torch.unsqueeze(torch.unsqueeze(bci_data3, 2),0)
        bci_data4 = torch.unsqueeze(torch.unsqueeze(bci_data4, 2),0)

        label = torch.unsqueeze(self.targt_gt[class_index][sample_index],0)
        return bci_data1, bci_data2, bci_data3, bci_data4, label

    def source_data_reader(self, source_index, class_index, sample_path):
        def simple_batch_norm_1d(x,dim):
            eps = 1e-5
            x_mean = torch.mean(x, dim=dim, keepdim=True)
            x_var = torch.mean((x - x_mean) ** 2, dim=dim, keepdim=True)
            x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
            return x_hat
        time_start = random.randint(35, int(self.sample_freq*4 + 35 - self.win_train))
        x1 = time_start
        x2 = time_start + self.win_train
        bci_data = np.load(os.path.join(self.source_path[source_index], 'train', class_index, sample_path))[:, ::self.down_sample][:, x1:x2]
        bci_data1 = iir_bandpass_filter(bci_data, 3, 14, self.sample_freq, 4).astype(np.float32)
        bci_data2 = iir_bandpass_filter(bci_data, 9, 26, self.sample_freq, 4).astype(np.float32)
        bci_data3 = iir_bandpass_filter(bci_data, 14, 38, self.sample_freq, 4).astype(np.float32)
        bci_data4 = iir_bandpass_filter(bci_data, 19, 50, self.sample_freq, 4).astype(np.float32)

        bci_data1 = torch.from_numpy(bci_data1.T).to(self.device)
        bci_data2 = torch.from_numpy(bci_data2.T).to(self.device)
        bci_data3 = torch.from_numpy(bci_data3.T).to(self.device)
        bci_data4 = torch.from_numpy(bci_data4.T).to(self.device)
        if self.transform is None:
            bci_data1 = simple_batch_norm_1d(bci_data1,dim=1)
            bci_data2 = simple_batch_norm_1d(bci_data2,dim=1)
            bci_data3 = simple_batch_norm_1d(bci_data3,dim=1)
            bci_data4 = simple_batch_norm_1d(bci_data4,dim=1)
        bci_data1 = torch.unsqueeze(torch.unsqueeze(bci_data1, 2),0)
        bci_data2 = torch.unsqueeze(torch.unsqueeze(bci_data2, 2),0)
        bci_data3 = torch.unsqueeze(torch.unsqueeze(bci_data3, 2),0)
        bci_data4 = torch.unsqueeze(torch.unsqueeze(bci_data4, 2),0)

        label = torch.from_numpy(np.asarray([float(class_index)-1])).to(self.device)
        return bci_data1, bci_data2, bci_data3, bci_data4, label

    def class_aware_iter(self):
        former_batch_size = self.batch_size
        for m in range(self.whole_iter):
            if m == self.iter:
                self.batch_size = self.min_target_length - self.iter * self.batch_size
            target_data1, target_data2, target_data3, target_data4, target_label = [], [], [], [], []
            source_data_ = {}
            for t in range(len(self.source_path)):
                source_data_[t] = [[],[],[],[],[]]
            for i in self.class_list:
                if m == self.iter:
                    for j in range(m * former_batch_size, self.min_target_length):
                        bci_data1, bci_data2, bci_data3, bci_data4, label = self.target_data_reader(i, j)
                        target_data1.append(bci_data1)
                        target_data2.append(bci_data2)
                        target_data3.append(bci_data3)
                        target_data4.append(bci_data4)
                        target_label.append(label)
                else:
                    for j in range(m * self.batch_size, (m + 1) * self.batch_size):
                        bci_data1, bci_data2, bci_data3, bci_data4, label = self.target_data_reader(i, j)
                        target_data1.append(bci_data1)
                        target_data2.append(bci_data2)
                        target_data3.append(bci_data3)
                        target_data4.append(bci_data4)
                        target_label.append(label)

                for s_ind in range(len(self.source_path)):
                    sample_list = random.sample(self.source_list[s_ind][int(i) - 1], self.batch_size)
                    for j in range(self.batch_size):
                        bci_data1, bci_data2, bci_data3, bci_data4, label = self.source_data_reader(s_ind, i,sample_list[j])
                        source_data_[s_ind][0].append(bci_data1)
                        source_data_[s_ind][1].append(bci_data2)
                        source_data_[s_ind][2].append(bci_data3)
                        source_data_[s_ind][3].append(bci_data4)
                        source_data_[s_ind][4].append(label)
            source_data = [[]]*len(self.source_path)
            target_data1 = torch.cat(target_data1, 0)
            target_data2 = torch.cat(target_data2, 0)
            target_data3 = torch.cat(target_data3, 0)
            target_data4 = torch.cat(target_data4, 0)
            target_label = torch.cat(target_label, 0)
            for s_ind in range(len(self.source_path)):
                source_data[s_ind].append(torch.cat(source_data_[s_ind][0], 0))
                source_data[s_ind].append(torch.cat(source_data_[s_ind][1], 0))
                source_data[s_ind].append(torch.cat(source_data_[s_ind][2], 0))
                source_data[s_ind].append(torch.cat(source_data_[s_ind][3], 0))
                source_data[s_ind].append(torch.cat(source_data_[s_ind][4], 0))
            yield source_data, [target_data1, target_data2, target_data3, target_data4, target_label]


class ClassAwareDataLoader(object):
    def __init__(self, source_path,
                 win_train,down_sample,sample_freq, batch_size,transform=None,device='cuda',
                 target_paths={}, classnames=[], target_gt=[], phase=None,
                 class_set=[], num_selected_classes=0,
                 seed=None, num_workers=0, drop_last=False, **kwargs):
        self.source_path = source_path
        self.target_paths = target_paths
        self.target_gt = target_gt
        self.win_train = win_train
        self.down_sample = down_sample
        self.sample_freq = sample_freq
        self.transform = transform
        self.device = device
        self.classnames = classnames
        self.class_set = class_set
        self.phase = phase
        self.seed = seed

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.kwargs = kwargs

    def construct(self):
        self.dataset = DatasetProcessing(self.target_paths, self.source_path, self.target_gt,
                                         win_train=self.win_train, down_sample=self.down_sample, sample_freq=self.sample_freq, transform=self.transform, device=self.device)
        self.data_iter = self.dataset.class_aware_iter()
        self.whole_iter = self.dataset.whole_iter
    def dataloader(self):
        return self.data_iter





