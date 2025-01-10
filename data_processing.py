import scipy.io as sio
import os
import random
import numpy as np
import shutil
import math


phase_list = ['train', 'test']
data_path = './data'
processed_data_path = './processed_ssvep_data/'
os.mkdir(processed_data_path)
sess_list = os.listdir(data_path)
print(sess_list)
label_list = [1, 2, 3, 4]
for sess_index, sess in enumerate(sess_list):
    print(sess)
    os.mkdir(os.path.join(processed_data_path, sess))
    sub_list = os.listdir(os.path.join(data_path, sess))
    for index, sub in enumerate(sub_list):
        print(sub)
        os.mkdir(os.path.join(processed_data_path, sess, sub))
        os.mkdir(os.path.join(processed_data_path, sess, sub, 'valid'))
        for phase in phase_list:
            os.mkdir(os.path.join(processed_data_path, sess, sub, phase))

            ori_data = sio.loadmat(os.path.join(data_path, sess, sub,
                                         "sess{}_subj{}_EEG_SSVEP.mat".format(str(sess_index + 1).zfill(2),
                                                                              sub[1:].zfill(2))),
                            variable_names=["EEG_SSVEP_{}".format(phase)])['EEG_SSVEP_{}'.format(phase)]
            data = ori_data['x'][0][0]
            label_array = ori_data['y_dec'][0][0][0]
            split_time = ori_data['t'][0][0][0]
            for label in label_list:
                os.mkdir(os.path.join(processed_data_path, sess, sub, phase, str(label)))
                ssvep_label_index = np.where(label_array == label)[0]
                for i in range(ssvep_label_index.shape[0]):
                    start_time_index = ssvep_label_index[i]
                    start_time = split_time[start_time_index]
                    if start_time == split_time[-1]:
                        stop_time = data.shape[0]
                    else:
                        stop_time = split_time[start_time_index + 1] + 1
                    trail_bci_data = data[start_time:stop_time, :].T.astype(np.float32)
                    np.save(os.path.join(processed_data_path, sess, sub, phase, str(label),
                                         'sess{}_{}_{}_{}_{}.npy'.format(str(sess_index + 1).zfill(2), phase, sub,
                                                                         str(label), str(i))),
                            trail_bci_data)

                if phase == 'train':
                    os.mkdir(os.path.join(processed_data_path, sess, sub, 'valid', str(label)))
                    full_train_list = os.listdir(os.path.join(processed_data_path, sess, sub, phase, str(label)))
                    valid_list = random.sample(full_train_list, math.ceil(0.1 * len(full_train_list)))
                    for i in range(len(valid_list)):
                        shutil.move(os.path.join(processed_data_path, sess, sub, phase, str(label), valid_list[i]),
                                    os.path.join(processed_data_path, sess, sub, 'valid', str(label)))

