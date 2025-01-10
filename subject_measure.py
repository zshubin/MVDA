import numpy as np
import os
import scipy.signal as sp
from scipy.signal import butter, lfilter
import numpy.fft as fft
from munkres import Munkres
import math
from sklearn.cross_decomposition import CCA
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


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
        y[i, :] = sp.filtfilt(b, a, data[i, :])
    return y


def psd_transform(data):
    complex_array = fft.fft(data)
    power = np.abs(complex_array) ** 2
    Fs = 1000
    T = 1 / Fs
    L = data.shape[-1]
    t = np.array([i * T for i in range(L)])
    freqs = fft.fftfreq(t.size, t[1] - t[0])
    psd_feature = power[:,freqs > 0]
    return psd_feature


def load_subject_data(sess, sub):
    bci_data_path = os.path.join('./processed_ssvep_data/', sess, sub, 'train')
    class_list = os.listdir(bci_data_path)
    data_list = []
    label_list = []
    x1 = 35
    x2 = 4*1000-35
    for class_index in range(len(class_list)):
        bci_file_name = os.listdir(os.path.join(bci_data_path,str(class_list[class_index])))
        class_bci_data = []
        label_list.append(np.ones((len(bci_file_name)))*(int(class_list[class_index])-1))
        for index in range(len(bci_file_name)):
            c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
            bci_data = np.load(os.path.join(bci_data_path, str(class_list[class_index]), bci_file_name[index]))[c, x1:x2]
            bci_data = iir_bandpass_filter(bci_data, 3, 50, 1000, 4).astype(np.float32)
            psd_bci_data = np.expand_dims(bci_data,axis=0)
            class_bci_data.append(psd_bci_data)
        class_bci_data = np.concatenate(class_bci_data,axis=0)
        data_list.append(class_bci_data)
    label = np.concatenate(label_list,axis=0).astype(np.int64)
    return data_list, label


def cosine_Matrix(_matrixA, _matrixB):
    _matrixA_matrixB = np.dot(_matrixA, _matrixB.transpose())
    _matrixA_norm = np.sqrt(np.multiply(_matrixA, _matrixA).sum(axis=1))
    _matrixB_norm = np.sqrt(np.multiply(_matrixB, _matrixB).sum(axis=1))
    return np.divide(_matrixA_matrixB, _matrixA_norm * _matrixB_norm.transpose())


def cca_dis_measure(_vec1, _vec2):
    dist = []
    n_components =1
    for i in range(_vec2.shape[0]):
        channel_dist = 0.
        cca = CCA(n_components)
        cca.fit(_vec1.T, _vec2[i, :, :].T)
        O1_a, O1_b = cca.transform(_vec1.T,_vec2[i, :, :].T)
        for ind_val in range(0, 1):
            channel_dist += np.abs(np.corrcoef(O1_a[:, ind_val], O1_b[:, ind_val])[0, 1])
        dist.append(channel_dist)
    dist = np.expand_dims(np.asarray(dist), 0)
    return dist


def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def k_means(samples, num_clusters, inti_centers=False, stop_epsion=1e-2, max_iter=100, seed=None):
    np.random.seed(2022) if seed is None else np.random.seed(seed)
    sample_cluster_index = np.zeros(samples.shape[0], dtype=np.int)
    if inti_centers:
        cluster_loc = inti_centers
    else:
        random_indices = np.arange(samples.shape[0], dtype=np.int)
        np.random.shuffle(random_indices)
        cluster_loc = samples[random_indices[:num_clusters],:,:]
    old_distance_var = -10000
    for itr in range(max_iter):
        sample_cluster_distance = [cca_dis_measure(samples[i,:,:], cluster_loc) for i in range(samples.shape[0])]
        sample_cluster_distance = np.concatenate(sample_cluster_distance, axis=0)
        sample_cluster_index = np.argmax(sample_cluster_distance, axis=1)
        avg_distance_var = 0
        for i in range(num_clusters):
            cluster_i = samples[sample_cluster_index == i]
            cluster_loc[i, :, :] = np.mean(cluster_i, axis=0)
            avg_distance_var += np.sum([cca_dis_measure(cluster_i[j,:, :], np.expand_dims(cluster_loc[i,:,:], 0)) for j in range(cluster_i.shape[0])])
        avg_distance_var/=num_clusters
        if np.abs(avg_distance_var - old_distance_var) < stop_epsion:
            break
        print("Itr %d, avg. distance variance: %f" % (itr, avg_distance_var))
        old_distance_var = avg_distance_var
    return cluster_loc, sample_cluster_index, avg_distance_var


def subject_sim_measure(sess, target_subject, source_subject, cluster_iter, stop_epsion, num_classes, top_number):
    target_data_list, label = load_subject_data(sess, target_subject)
    target_data = np.concatenate(target_data_list, axis=0)
    centers, cluster_label, dis = k_means(target_data, num_classes, max_iter=cluster_iter, stop_epsion=stop_epsion)
    # new_y = best_map(label, cluster_label)
    # acc = np.sum(new_y == label) / cluster_label.shape[0]
    # print(label)
    # print(cluster_label)
    pur = purity(label,cluster_label)
    coff_list = []
    for sub_i in range(len(source_subject)):
        source_data_list,label = load_subject_data(sess, source_subject[sub_i])
        dist_matrix = []
        for i in range(len(source_data_list)):
            class_mean = np.mean(source_data_list[i],axis=0)
            corr_s2t = cca_dis_measure(class_mean,centers)
            dist_matrix.append(corr_s2t)
        dist_matrix = np.concatenate(dist_matrix,axis=0)
        m = Munkres()
        index = m.compute(-dist_matrix)
        index = np.array(index)
        max_coff = 0.
        for i in range(centers.shape[0]):
            max_coff += dist_matrix[index[i,0],index[i,1]]
        coff_list.append(max_coff)
    sorted_id = sorted(range(len(coff_list)), key=lambda x: coff_list[x], reverse=True)
    top_sub = []
    top_coff = []
    for i in range(top_number):
        top_sub.append(source_subject[sorted_id[i]])
        top_coff.append(coff_list[sorted_id[i]])
    return pur, dis/22,top_sub,top_coff


if __name__ == '__main__':
    sess = 'session1'
    subject_list = os.listdir(os.path.join('./processed_ssvep_data/', sess))
    subject_list.remove('s23')
    subject_list.remove('s47')
    target_subject_list = ['s1', 's5', 's7', 's9', 's11', 's12', 's16', 's17', 's19', 's29', 's31', 's36', 's39',
                           's42','s43', 's44', 's45', 's49', 's53']
    for sub in target_subject_list:
        subject_list.remove(sub)
    sub_list = []
    mean_dist = []
    coff_list = []
    acc_list = []
    for i in range(len(target_subject_list)):
        target_subject = target_subject_list[i]
        cluster_iter = 50
        stop_epsion = 1e-10
        num_classes = 4
        top_number = 20
        acc,dis,sub,top_coff = subject_sim_measure(sess, target_subject, subject_list, cluster_iter, stop_epsion,
                                                num_classes, top_number)
        sub_list.append(sub[:4])
        mean_dist.append(dis)
        coff_list.append(top_coff[:4])
        acc_list.append(acc)
    print(acc_list)
    print(mean_dist)
    print(sub_list)
    print(coff_list)


