import numpy as np
import os
from data_loader_cross_sub.DataProcessing import Learner, Filter, my_cca, my_cca1
import warnings
import scipy.signal as sp
from sklearn.cross_decomposition import CCA
from munkres import Munkres
warnings.filterwarnings('ignore')

def iir_bandpass_filter(data, lowcut=8, highcut=30, samplingRate=512, order=4):
    y = np.zeros_like(data).astype(np.float32)
    nyq = 0.5 * samplingRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp.iirfilter(order, [low, high], btype='band')
    for i in range(data.shape[0]):
        y[i, :] = sp.filtfilt(b, a, data[i, :])
    return y


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


def get_cca_reference_signals(data_len, target_freq, sampling_rate, harmo_num):
    reference_signals = []
    t = np.arange(0, (data_len / (sampling_rate)), step=1.0 / (sampling_rate))
    for i in range(harmo_num):
        reference_signals.append(np.sin(np.pi * 2 * (i+1) * target_freq * t))
        reference_signals.append(np.cos(np.pi * 2 * (i+1) * target_freq * t))
    reference_signals = np.array(reference_signals)[:, ::down_sample]
    return reference_signals


def ECCA_fit_transform(data_mean_list, freq_list, epoch_data_test, epoch_label_test):
    label_list = sorted(list(set(epoch_label_test)))
    epoch_label_test = np.squeeze(np.array(epoch_label_test))

    predicted_result = np.zeros(epoch_data_test.shape[0])
    coeff_list = []
    for j in range(epoch_data_test.shape[0]):
        result = np.zeros((1,len(label_list)))
        epoch_data = epoch_data_test[j, :, :]
        for freq_idx in range(len(label_list)):
            coeff = np.zeros(4)
            ref_data = freq_list[freq_idx]
            # rho 1
            wn1, wn2 = my_cca1(epoch_data, ref_data)
            weighted_train = np.dot(wn2.T, ref_data)
            weighted_test = np.dot(wn1.T, epoch_data)
            coeff[0] = np.abs(np.corrcoef(weighted_test, weighted_train)[0, 1])
            # rho 2
            wn, _ = my_cca1(epoch_data, data_mean_list[freq_idx])
            weighted_train = np.dot(wn.T, data_mean_list[freq_idx])
            weighted_test = np.dot(wn.T, epoch_data)
            coeff[1] = np.corrcoef(weighted_test, weighted_train)[0, 1]
            # rho 3
            wn, _ = my_cca1(epoch_data, ref_data)
            weighted_train = np.dot(wn.T, data_mean_list[freq_idx])
            weighted_test = np.dot(wn.T, epoch_data)
            coeff[2] = np.corrcoef(weighted_test, weighted_train)[0, 1]
            # rho 4
            wn, _ = my_cca1(data_mean_list[freq_idx], ref_data)
            weighted_train = np.dot(wn.T, data_mean_list[freq_idx])
            weighted_test = np.dot(wn.T, epoch_data)
            coeff[3] = np.corrcoef(weighted_test, weighted_train)[0, 1]
            result[0,freq_idx] = np.sum(np.sign(coeff) * np.square(coeff))
        coeff_list.append(result)
        predicted_result[j] = np.argmax(result)
    return predicted_result, coeff_list


def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort


sess = 'session1'
down_sample = 4
SAMPLING_FREQ = 1000/down_sample
LOWER_CUTOFF = 3.
UPPER_CUTTOF = 50.
filt_type = 'iir'
win_type = None
FILT_ORDER = 4
cluster_iter = 1000
freq_list = [5.45, 12, 8.57, 6.67]
stop_epsion = 1e-10
top_number = 20
threshold = 1.15
alpha = 0.07
source_num = 2
c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
# time_win_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
time_win_list = [1.]
for time_win in time_win_list:
    record_file = open('./record_adaptivec3a_3fold_5source.txt'.format(str(time_win)), 'w')
    sine_consine_temp = [get_cca_reference_signals(int(time_win * 1000), i, 1000, 5) for i in freq_list]
    subject_list = os.listdir(os.path.join('./processed_ssvep_data/', sess))
    subject_list.remove('s23')
    subject_list.remove('s47')
    target_list = ['s16', 's19', 's30', 's48', 's53']
    for sub in target_list:
        subject_list.remove(sub)

    acc_list = []
    for sub in target_list:
        tar_sub = sub
        top_sub, top_coff = subject_sim_measure(sess, sub, subject_list, cluster_iter, stop_epsion, len(freq_list),top_number)
        source_list = top_sub[:source_num]
        full_path = os.path.join('./processed_ssvep_data/', sess, sub)
        dp = Filter(LOWER_CUTOFF, UPPER_CUTTOF, SAMPLING_FREQ, FILT_ORDER, filt_type=filt_type, win_type=win_type)
        ################################train_1##############################
        train_list = os.listdir(os.path.join(full_path, 'train', '1'))
        bci_data = np.load(os.path.join(full_path, 'train', '1', train_list[0]))[:, ::down_sample][:,
                   int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
        bci_data = np.expand_dims(bci_data, axis=0)
        for i in range(len(train_list) - 1):
            mini_bci_data = np.load(os.path.join(full_path, 'train', '1', train_list[i + 1]))[:, ::down_sample][:,
                            int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            mini_bci_data = dp.ApplyFilter(mini_bci_data)
            mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
            bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
        train_label_cal = [0.] * len(train_list)
        #####################################train_2##########################
        train_list = os.listdir(os.path.join(full_path, 'train', '2'))
        for i in range(len(train_list)):
            mini_bci_data = np.load(os.path.join(full_path, 'train', '2', train_list[i]))[:, ::down_sample][:,
                            int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            mini_bci_data = dp.ApplyFilter(mini_bci_data)
            mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
            bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
        labels_cal = [1.] * len(train_list)
        train_label_cal.extend(labels_cal)
        ################################train_3###############################
        train_list = os.listdir(os.path.join(full_path, 'train', '3'))
        for i in range(len(train_list)):
            mini_bci_data = np.load(os.path.join(full_path, 'train', '3', train_list[i]))[:, ::down_sample][:,
                            int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            mini_bci_data = dp.ApplyFilter(mini_bci_data)
            mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
            bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
        labels_cal = [2.] * len(train_list)
        train_label_cal.extend(labels_cal)
        #####################################train_4##########################
        train_list = os.listdir(os.path.join(full_path, 'train', '4'))
        for i in range(len(train_list)):
            mini_bci_data = np.load(os.path.join(full_path, 'train', '4', train_list[i]))[:, ::down_sample][:,
                            int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            mini_bci_data = dp.ApplyFilter(mini_bci_data)
            mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
            bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
        labels_cal = [3.] * len(train_list)
        train_label_cal.extend(labels_cal)
        train_data_cal = bci_data
        #######################################valid_1########################
        val_list = os.listdir(os.path.join(full_path, 'valid', '1'))
        bci_data = np.load(os.path.join(full_path, 'valid', '1', val_list[0]))[:, ::down_sample][:,
                   int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
        bci_data = np.expand_dims(bci_data, axis=0)
        for i in range(len(val_list) - 1):
            mini_bci_data = np.load(os.path.join(full_path, 'valid', '1', val_list[i + 1]))[:, ::down_sample][:,
                            int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            mini_bci_data = dp.ApplyFilter(mini_bci_data)
            mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
            bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
        val_label_cal = [0.] * len(val_list)
        #################################valid_2##############################
        val_list = os.listdir(os.path.join(full_path, 'valid', '2'))
        for i in range(len(val_list)):
            mini_bci_data = np.load(os.path.join(full_path, 'valid', '2', val_list[i]))[:, ::down_sample][:,
                            int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            mini_bci_data = dp.ApplyFilter(mini_bci_data)
            mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
            bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
        val_label_cal.extend([1.] * len(val_list))
        #################################valid_3##############################
        val_list = os.listdir(os.path.join(full_path, 'valid', '3'))
        for i in range(len(val_list)):
            mini_bci_data = np.load(os.path.join(full_path, 'valid', '3', val_list[i]))[:, ::down_sample][:,
                            int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            mini_bci_data = dp.ApplyFilter(mini_bci_data)
            mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
            bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
        val_label_cal.extend([2.] * len(val_list))
        #################################valid_4##############################
        val_list = os.listdir(os.path.join(full_path, 'valid', '4'))
        for i in range(len(val_list)):
            mini_bci_data = np.load(os.path.join(full_path, 'valid', '4', val_list[i]))[:, ::down_sample][:,
                            int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            mini_bci_data = dp.ApplyFilter(mini_bci_data)
            mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
            bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
        val_label_cal.extend([3.] * len(val_list))
        val_data_cal = bci_data

        class_1_list, class_2_list, class_3_list, class_4_list = [], [], [], []
        #################source_template##########
        for j in range(len(source_list)):
            sub = source_list[j]
            print(sub)
            ################################train_1################################
            train_list = os.listdir(os.path.join(full_path, 'train', '1'))
            bci_data = np.load(os.path.join(full_path, 'train', '1', train_list[0]))[:, ::down_sample][:,
                       int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            bci_data = np.expand_dims(bci_data, axis=0)
            for i in range(len(train_list) - 1):
                mini_bci_data = np.load(os.path.join(full_path, 'train', '1', train_list[i + 1]))[:, ::down_sample][:,
                                int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
                mini_bci_data = dp.ApplyFilter(mini_bci_data)
                mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
                bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
            class_1_list.append(bci_data)
            #####################################train_2###########################
            train_list = os.listdir(os.path.join(full_path, 'train', '2'))
            bci_data = np.load(os.path.join(full_path, 'train', '2', train_list[0]))[:, ::down_sample][:,
                       int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            bci_data = np.expand_dims(bci_data, axis=0)
            for i in range(len(train_list) - 1):
                mini_bci_data = np.load(os.path.join(full_path, 'train', '2', train_list[i + 1]))[:, ::down_sample][:,
                                int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
                mini_bci_data = dp.ApplyFilter(mini_bci_data)
                mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
                bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
            class_2_list.append(bci_data)
            ################################train_3################################
            train_list = os.listdir(os.path.join(full_path, 'train', '3'))
            bci_data = np.load(os.path.join(full_path, 'train', '3', train_list[0]))[:, ::down_sample][:,
                       int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            bci_data = np.expand_dims(bci_data, axis=0)
            for i in range(len(train_list) - 1):
                mini_bci_data = np.load(os.path.join(full_path, 'train', '3', train_list[i + 1]))[:, ::down_sample][:,
                                int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
                mini_bci_data = dp.ApplyFilter(mini_bci_data)
                mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
                bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
            class_3_list.append(bci_data)
            #####################################train_4###########################
            train_list = os.listdir(os.path.join(full_path, 'train', '4'))
            bci_data = np.load(os.path.join(full_path, 'train', '4', train_list[0]))[:, ::down_sample][:,
                       int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
            bci_data = np.expand_dims(bci_data, axis=0)
            for i in range(len(train_list) - 1):
                mini_bci_data = np.load(os.path.join(full_path, 'train', '4', train_list[i + 1]))[:, ::down_sample][:,
                                int(SAMPLING_FREQ * (0.14)): int(SAMPLING_FREQ * (time_win + 0.14))]
                mini_bci_data = dp.ApplyFilter(mini_bci_data)
                mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
                bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
            class_4_list.append(bci_data)
        class_1_data, class_2_data, class_3_data, class_4_data = np.concatenate(class_1_list, axis=0), np.concatenate(
            class_2_list, axis=0), np.concatenate(class_3_list, axis=0), np.concatenate(class_4_list, axis=0)
        class_1_template, class_2_template, class_3_template, class_4_template = np.mean(class_1_data, axis=0), np.mean(
            class_2_data, axis=0), np.mean(class_3_data, axis=0), np.mean(class_4_data, axis=0)
        class_template = [class_1_template, class_2_template, class_3_template, class_4_template]

        _, coeff_list = ECCA_fit_transform(class_template, sine_consine_temp, train_data_cal, train_label_cal)
        coeff_ = np.concatenate(coeff_list,axis=0)
        sample_index = np.arange(coeff_.shape[0])
        top2_data, top2_ind = topk_(coeff_, 2, 1)
        BvSB = top2_data[:,0]/top2_data[:,1]
        filtered_sample_index = sample_index[BvSB > threshold]
        filtered_label = top2_ind[filtered_sample_index, 0]
        for i in range(len(freq_list)):
            additional_sample = filtered_sample_index[filtered_label == int(i)]
            if additional_sample.shape[0] == 0:
                print("Zero sample")
                continue
            else:
                for j in range(additional_sample.shape[0]):
                    class_template[i] += alpha * additional_sample[j]

        predicted_result, coeff_list = ECCA_fit_transform(class_template, sine_consine_temp, val_data_cal,
                                                          val_label_cal)
        acc = np.sum(predicted_result == val_label_cal) / val_data_cal.shape[0]
        acc_list.append(acc)
        record_file.write(tar_sub + ' ' + str(acc) + '\n')
    record_file.close()
