from torch.backends import cudnn
from DDGCNN_cross_sub_GMM_3sub.cdd_model import Extractor, Predictor
from DDGCNN_cross_sub_GMM_3sub.config import OptInit
import logging
from data_loader_cross_sub.prepare_data import *
from DDGCNN_cross_sub_GMM_3sub.subject_measure import subject_sim_measure
import sys
from DDGCNN_cross_sub_GMM_3sub.can_solver import CANSolver as Solver


def train(logger, sess, source_sub, tar_sub, rest_sub,record_file,top_coff):
    opt_class = OptInit(tar_sub, logger)
    opt = opt_class.get_args()
    data_path = []
    for i in source_sub:
        data_path.append(os.path.join('./processed_ssvep_data/', sess, i))
    data_path.append(os.path.join('./processed_ssvep_data/', sess, tar_sub))
    rest_sub_path = []
    for i in rest_sub:
        rest_sub_path.append(os.path.join('./processed_ssvep_data/', sess, i))
    dataloaders = prepare_data_CAN(opt.batch_size,data_path,rest_sub_path,opt.time_win,opt.down_sample,opt.sample_freq,opt.device)

    Extractor_model = Extractor([opt.batch_size, opt.in_channels, opt.eeg_channel], opt.k_adj, opt.n_filters, opt.dropout,
                      bias=opt.bias, norm=opt.norm, act=opt.act, trans_class=opt.trans_class, device=opt.device).to(opt.device)

    Predictors = {}
    # Predictor_model = Predictor(opt.n_filters, opt.n_blocks, opt.class_num,).to(opt.device)
    para_source_sub = [i for i in source_sub]
    for source in para_source_sub:
        Predictors[source] = {}
        Predictors[source]['c1'] = Predictor(opt.n_filters, opt.n_blocks, opt.class_num, ).to(opt.device)
        Predictors[source]['c2'] = Predictor(opt.n_filters, opt.n_blocks, opt.class_num, ).to(opt.device)

    train_solver = Solver(Extractor_model, Predictors, record_file, top_coff=top_coff, dataloader=dataloaders,
                          device=opt.device,
                          num_layers=opt.num_layers, kernel_num=opt.kernel_num, kernel_mul=opt.kernel_mul,
                          num_classes=opt.class_num, intra_only=opt.intra_only, clustering_source_name=para_source_sub,
                          rest_sub_name=rest_sub,
                          clustering_target_name='clustering_target', logger=logger, save_dir=opt.save_dir, opt=opt)
    train_solver.solve()
    print('Finished!')


def make_logger():
    loglevel = "info"
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(loglevel))
    log_format = logging.Formatter('%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logging.root = logger
    return logger


if __name__ == '__main__':
    cudnn.benchmark = True
    logger = make_logger()
    source_num = 4
    sess = 'session1'
    subject_list = os.listdir(os.path.join('./processed_ssvep_data/', sess))
    subject_list.remove('s23')
    subject_list.remove('s47')
    target_subject_list = ['s1', 's3', 's5', 's7', 's9', 's11', 's12', 's16', 's17', 's19', 's29', 's31', 's36', 's39', 's42',
                's43', 's44', 's45', 's49', 's53']
    for sub in target_subject_list:
        subject_list.remove(sub)
    record_file = open('./record_3_file0.9.txt','w')
    for i in range(len(target_subject_list)):
        target_subject = target_subject_list[i]
        cluster_iter = 1000
        stop_epsion = 1e-10
        num_classes = 4
        top_number = 20
        top_sub, top_coff = subject_sim_measure(sess, target_subject, subject_list, cluster_iter, stop_epsion, num_classes,top_number)
        source_subject_list = top_sub[:source_num]
        record_file.write('target' + ' ' + target_subject + ' ' + 'source' + ' '
                          + source_subject_list[0] + ' '
                          + source_subject_list[1] + ' '
                          + source_subject_list[2] + ' '
                          + source_subject_list[3] + ' '
                          +'\n')
        rest_subject_list = top_sub[4:9]
        print(rest_subject_list)
        train(logger, sess, source_subject_list, target_subject, rest_subject_list, record_file, top_coff)
    record_file.close()

