from data_loader_cross_sub.class_aware_dataloader import ClassAwareDataLoader
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
import scipy.signal as sp
from scipy.signal import butter, lfilter


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
    def __init__(self, data_path, phase, win_train=1, down_sample=1, sample_freq=1000, transform=None, device='cuda'):
        self.bci_data_path = os.path.join(data_path, phase) #data/train data/test
        self.transform = transform
        self.bci_file_name = []
        self.sample_freq = int(sample_freq/down_sample)
        self.win_train = int(self.sample_freq*win_train)
        self.phase = phase
        self.down_sample = down_sample
        self.label = []
        self.device = device
        class_num = 0.
        self.class_dict = {}
        for class_name in os.listdir(self.bci_data_path):
            class_bci_file = os.listdir(os.path.join(self.bci_data_path, class_name))
            self.bci_file_name.extend(class_bci_file)
            self.label.extend([class_num]*len(class_bci_file))
            self.class_dict[class_num] = class_name
            class_num += 1.
        self.label = np.array(self.label).astype(np.float32)

    def __getitem__(self, index):

        def simple_batch_norm_1d(x,dim):
            eps = 1e-5
            x_mean = torch.mean(x, dim=dim, keepdim=True)
            x_var = torch.mean((x - x_mean) ** 2, dim=dim, keepdim=True)
            x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
            return x_hat
        label_name = self.class_dict[self.label[index]]
        time_start = random.randint(35, int(self.sample_freq*4 + 35 - self.win_train))
        x1 = time_start
        x2 = time_start + self.win_train
        bci_data = np.load(os.path.join(self.bci_data_path, label_name, self.bci_file_name[index]))[:, ::self.down_sample][:, x1:x2]
        bci_data1 = iir_bandpass_filter(bci_data, 3, 14, self.sample_freq, 4).astype(np.float32)
        bci_data2 = iir_bandpass_filter(bci_data, 9, 26, self.sample_freq, 4).astype(np.float32)
        bci_data3 = iir_bandpass_filter(bci_data, 14, 38, self.sample_freq, 4).astype(np.float32)
        bci_data4 = iir_bandpass_filter(bci_data, 19, 50, self.sample_freq, 4).astype(np.float32)

        bci_data1 = torch.from_numpy(bci_data1.T)#.to(self.device)
        bci_data2 = torch.from_numpy(bci_data2.T)#.to(self.device)
        bci_data3 = torch.from_numpy(bci_data3.T)#.to(self.device)
        bci_data4 = torch.from_numpy(bci_data4.T)#.to(self.device)
        if self.transform is None:
            bci_data1 = simple_batch_norm_1d(bci_data1,dim=1)
            bci_data2 = simple_batch_norm_1d(bci_data2,dim=1)
            bci_data3 = simple_batch_norm_1d(bci_data3,dim=1)
            bci_data4 = simple_batch_norm_1d(bci_data4,dim=1)
        bci_data1 = torch.unsqueeze(bci_data1, 2)
        bci_data2 = torch.unsqueeze(bci_data2, 2)
        bci_data3 = torch.unsqueeze(bci_data3, 2)
        bci_data4 = torch.unsqueeze(bci_data4, 2)

        label = torch.from_numpy(np.array(self.label[index])).to(self.device)
        return [bci_data1, bci_data2, bci_data3, bci_data4, label, os.path.join(self.bci_data_path, label_name, self.bci_file_name[index])]

    def __len__(self):
        return len(self.bci_file_name)

def prepare_data_CAN(batch_size,data_path,rest_path,win_train,down_sample,sample_freq,device):
    dataloaders = {}
    source_number = len(data_path)-1
    # for clustering
    #################################
    for i in range(source_number):
        train_dataset = DatasetProcessing(data_path[i], 'train', win_train=win_train, down_sample=down_sample,
                                      sample_freq=sample_freq, device=device)
        dataloaders['clustering_s{}'.format(i+1)] = torch.utils.data.DataLoader(dataset=train_dataset,
                                                                   batch_size=batch_size,
                                                                   shuffle=True,
                                                                   drop_last=False,
                                                                   num_workers=0)

    target_train_dataset = DatasetProcessing(data_path[-1], 'train', win_train=win_train, down_sample=down_sample,
                                   sample_freq=sample_freq, device=device)
    dataloaders['clustering_target'] = torch.utils.data.DataLoader(dataset=target_train_dataset,
                                                                      batch_size=batch_size,
                                                                      shuffle=True,
                                                                      drop_last=False,
                                                                      num_workers=0)
    ###################
    for i in range(source_number):
        test_dataset = DatasetProcessing(data_path[i], 'valid', win_train=win_train, down_sample=down_sample,
                                      sample_freq=sample_freq, device=device)
        dataloaders['s{}_valid'.format(i+1)] = torch.utils.data.DataLoader(dataset=test_dataset,
                                                                   batch_size=batch_size,
                                                                   shuffle=True,
                                                                   drop_last=False,
                                                                   num_workers=0)


    target_test_dataset = DatasetProcessing(data_path[-1], 'train', win_train=win_train, down_sample=down_sample,
                                   sample_freq=sample_freq, device=device)
    dataloaders['target_valid'] = torch.utils.data.DataLoader(dataset=target_test_dataset,
                                                                      batch_size=batch_size,
                                                                      shuffle=True,
                                                                      drop_last=False,
                                                                      num_workers=0)
    # for i in range(len(rest_path)):
    #     test_dataset = DatasetProcessing(rest_path[i], 'test', win_train=win_train, down_sample=down_sample,
    #                                      sample_freq=sample_freq, device=device)
    #     dataloaders['rest_{}'.format(rest_path[i])] = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                                                         batch_size=batch_size,
    #                                                                         shuffle=True,
    #                                                                         drop_last=False,
    #                                                                         num_workers=0)

    source_batch_size = batch_size
    dataloaders['target_categorical'] = ClassAwareDataLoader(
        source_path = data_path[:source_number], batch_size=source_batch_size,
        win_train=win_train,
        down_sample=down_sample,
        sample_freq=sample_freq,
        device=device,
        classnames=['1', '2', '3', '4'],
        num_workers=0,
        drop_last=False)

    return dataloaders
