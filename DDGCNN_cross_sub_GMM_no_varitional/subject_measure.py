import numpy as np
import os
import scipy.signal as sp
from scipy.signal import butter, lfilter
import numpy.fft as fft
from munkres import Munkres
import math
from sklearn.cross_decomposition import CCA


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
    x1 = 35
    x2 = 4*1000-35
    for class_index in range(len(class_list)):
        bci_file_name = os.listdir(os.path.join(bci_data_path,str(class_list[class_index])))
        class_bci_data = []
        for index in range(len(bci_file_name)):
            c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
            bci_data = np.load(os.path.join(bci_data_path, str(class_list[class_index]), bci_file_name[index]))[c, x1:x2]
            bci_data = iir_bandpass_filter(bci_data, 3, 50, 1000, 4).astype(np.float32)
            psd_bci_data = np.expand_dims(bci_data,axis=0)
            class_bci_data.append(psd_bci_data)
        class_bci_data = np.concatenate(class_bci_data,axis=0)
        data_list.append(class_bci_data)
    return data_list


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
    return cluster_loc, sample_cluster_index

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

def cluster_probability(cluster):
    cluster_sum = []
    n = float(len(cluster))
    cluster_names = list(set(cluster))
    for index, cluster_name in enumerate(cluster_names):
        cluster_sum.append(0)
        for i in cluster:
            if (cluster_name == i):
                cluster_sum[index] += 1
    cluster_prob = [i / n for i in cluster_sum]
    return cluster_prob

def entropy(cluster):
    cluster_prob = cluster_probability(cluster)
    entropy = 0
    for i in cluster_prob:
        entropy += i * math.log(i, 2)
    return entropy * (-1)

def mutual_information(cluster, truth):
    n = float(len(cluster))
    pc = cluster_probability(cluster)
    pt = cluster_probability(truth)

    sc = list(set(cluster))  # name of the clusters
    st = list(set(truth))  # name of the clusters in ground truth

    p_matrix = [[0 for x in range(len(sc))] for y in range(len(st))]

    for i in range(len(truth)):
        p_matrix[st.index(truth[i])][sc.index(cluster[i])] += 1
    mutual_info = 0
    for i in range(len(st)):
        for j in range(len(sc)):
            joint_p = (p_matrix[i][j] / (n))
            if (joint_p != 0):
                mutual_info += joint_p * (math.log(joint_p / float(pc[j] * pt[i]), 2))
    return mutual_info

def nmi(cluster, truth):
    hc = entropy(cluster)
    ht = entropy(truth)
    mutual_info = mutual_information(cluster, truth)
    return mutual_info / math.sqrt(hc * ht)

def subject_sim_measure(sess, target_subject, source_subject, cluster_iter, stop_epsion, num_classes, top_number):
    target_data_list = load_subject_data(sess, target_subject)
    target_data = np.concatenate(target_data_list, axis=0)
    centers, cluster_label = k_means(target_data, num_classes, max_iter=cluster_iter, stop_epsion=stop_epsion)
    coff_list = []
    for sub_i in range(len(source_subject)):
        source_data_list = load_subject_data(sess, source_subject[sub_i])
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
    return top_sub, top_coff




if __name__ == '__main__':
    # sub_data = load_subject_data('session1','s23')
    # sub_label = [np.zeros(sub_data[0].shape[0]),np.ones(sub_data[0].shape[0]),np.ones(sub_data[0].shape[0])*2,np.ones(sub_data[0].shape[0])*3]
    # sub_label = np.concatenate(sub_label,axis=0)
    # sub_data = np.concatenate(sub_data,axis=0)
    # index = np.random.permutation(sub_data.shape[0])
    # sub_data = sub_data[index]
    # sub_label = sub_label[index]
    # centers, cluster_label = k_means(sub_data,4,max_iter=500,stop_epsion=1e-20)
    # pre_label = best_map(sub_label,cluster_label)
    # acc = np.sum(pre_label == sub_label)/sub_data.shape[0]
    # print(acc)
    cluster_iter = 1000
    stop_epsion  = 1e-10
    num_classes  = 4
    top_number   = 4
    top_sub = subject_sim_measure('session1','s50',['s13','s14','s15','s20','s23','s25'], cluster_iter, stop_epsion, num_classes, top_number)
    print(top_sub)


