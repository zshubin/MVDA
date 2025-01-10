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


def my_softmax(x):
    # exp_x = np.exp(x)
    # x= exp_x/np.sum(exp_x)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

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
source_num = 5
c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
# time_win_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
time_win_list = [1.]
for time_win in time_win_list:
    record_file = open('./record_ttcca_3fold_5source.txt', 'w')
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
        top_sub, top_coff = subject_sim_measure(sess, sub, subject_list, cluster_iter, stop_epsion, len(freq_list),
                                                top_number)
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
        #####################calculate_average_template_filter##########################
        Wx_list1 = []
        for i in range(len(class_template)):
            cca = CCA(3)
            cca.fit(class_template[i].T, sine_consine_temp[i].T)
            Wx, Wy = cca.x_rotations_, cca.y_rotations_
            # Wx, Wy, r = my_cca(class_template[i], sine_consine_temp[i])
            Wx_list1.append(np.reshape(Wx[:, 0], (-1, 1)))
        #####################calculate_target_data_filter###############################
        Wx_list3 = []
        for j in range(val_data_cal.shape[0]):
            sub_Wx_list = []
            for i in range(len(class_template)):
                cca = CCA(3)
                cca.fit(val_data_cal[j].T, sine_consine_temp[i].T)
                Wx, Wy = cca.x_rotations_, cca.y_rotations_
                # Wx, Wy, r = my_cca(val_data_cal[j], sine_consine_temp[i])
                sub_Wx_list.append(np.reshape(Wx[:, 0], (-1, 1)))
            Wx = np.concatenate(sub_Wx_list, axis=1)
            Wx_list3.append(Wx)
        #####################calculate_rho_1############################################
        rho_1_list = []
        for j in range(val_data_cal.shape[0]):
            sub_rho_1 = []
            for i in range(len(class_template)):
                weighted_data = np.dot(Wx_list1[i].T, val_data_cal[j])
                weighted_temp = np.dot(Wx_list1[i].T, class_template[i])
                sub_rho_1.append(np.corrcoef(weighted_temp, weighted_data)[0, 1])
            rho_1_list.append(np.asarray(sub_rho_1))
        #####################calculate_rho_2############################################
        rho_2_list = []
        for j in range(val_data_cal.shape[0]):
            sub_rho_2 = []
            for i in range(len(class_template)):
                cca = CCA(3)
                cca.fit(val_data_cal[j].T, sine_consine_temp[i].T)
                Wx, Wy = cca.x_rotations_, cca.y_rotations_
                # Wx, Wy, y = my_cca(val_data_cal[j], sine_consine_temp[i])
                weighted_data = np.dot(np.reshape(Wx[:, 0], (-1, 1)).T, val_data_cal[j])
                weighted_temp = np.dot(np.reshape(Wy[:, 0], (-1, 1)).T, sine_consine_temp[i])
                sub_rho_2.append(np.corrcoef(weighted_temp, weighted_data)[0, 1])
            rho_2_list.append(np.asarray(sub_rho_2))
        #####################calculate_rho_3############################################
        rho_3_list = []
        for j in range(val_data_cal.shape[0]):
            sub_rho_3 = []
            for i in range(len(class_template)):
                weighted_data = np.dot(Wx_list3[j][:, i].T, val_data_cal[j])
                weighted_temp = np.dot(Wx_list3[j][:, i].T, class_template[i])
                sub_rho_3.append(np.corrcoef(weighted_temp, weighted_data)[0, 1])
            rho_3_list.append(np.asarray(sub_rho_3))

        predicted_result = np.zeros(val_data_cal.shape[0])
        for i in range(val_data_cal.shape[0]):
            rho = 0.1 * rho_1_list[i] + 0.1 * rho_2_list[i] + 0.8 * rho_3_list[i]
            predicted_result[i] = np.argmax(rho)
        acc = np.sum(predicted_result == val_label_cal) / val_data_cal.shape[0]
        acc_list.append(acc)
        print(acc)
        record_file.write(tar_sub + ' ' + str(acc) + '\n')
    record_file.close()
    print(acc_list)

