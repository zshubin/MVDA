import argparse
import random
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
import os


class OptInit:
    def __init__(self, sub, logger):
        self.logger = logger
        parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN')

        # base
        parser.add_argument('--use_cpu', default=True, help='use cpu')

        # dataset args
        parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size (default:32)')
        parser.add_argument('--sample_freq', default=1000, type=int, help='bci sample frequency')
        parser.add_argument('--down_sample', default=4, type=int, help='down sample rate')
        parser.add_argument('--eeg_channel', type=int, default=62, help='eeg_channel')
        parser.add_argument('--class_num', type=int, default=4, help='ssvep class')
        parser.add_argument('--time_win', type=float, default=1., help='time window')

        # train args
        parser.add_argument('--total_epochs', default=5000, type=int, help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=2022, help='random seed')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
        parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
        parser.add_argument('--lr_decay_rate', default=0.5, type=float, help='learning rate decay')
        parser.add_argument('--optim_patience', default=400, type=int, help='learning rate decay patience epoch')
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum rate')
        parser.add_argument('--decay_step', default=30, type=float, help='lr decay step')
        parser.add_argument('--inv_alpha', default=0.001, type=float, help='inverse alpha')
        parser.add_argument('--inv_beta', default=0.75, type=float, help='inverse beta')
        parser.add_argument('--lr_schedule', default='plateau', type=str, help='learning rate schedule method:exp and inv, plateau')
        parser.add_argument('--stop_thresholds', default=(0.001, 0.001, 0.001), type=tuple, help='training threshold')
        parser.add_argument('--max_loop', default=10, type=int, help='max loop')

        # model args
        parser.add_argument('--trans_class', default='DCD', type=str, help='{DCD, linear, normal_conv}')
        parser.add_argument('--act', default='leakyrelu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm', default='layer', type=str, help='{batch, layer, instance} normalizaxtion')
        parser.add_argument('--bias', default=False,  type=bool, help='bias of conv layer True or False')
        parser.add_argument('--n_filters', default=32, type=int, help='number of channels of deep features')
        parser.add_argument('--k_adj', type=int, default=3, help='adj order')
        parser.add_argument('--n_blocks', default=1, type=int, help='number of basic blocks')
        parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')

        #CDD args
        parser.add_argument('--clustering_eps', default=1e-6, type=float, help='clustering threshold')
        parser.add_argument('--dist_type', default='cos', type=str, help='clustering distance type')
        parser.add_argument('--filtering_threshold', default=1e+4, type=int, help='threshold for clustering filtering')
        parser.add_argument('--min_sn_sample', default=4, type=int, help='min sample for target wise sample')
        parser.add_argument('--num_layers', default=1, type=int, help='CDD layer number')
        parser.add_argument('--kernel_num', default=(3, 3), type=tuple, help='kernel number')
        parser.add_argument('--kernel_mul', default=(2, 2), type=tuple, help='')
        parser.add_argument('--intra_only', default=False, type=bool, help='')
        parser.add_argument('--optim', default='Adam', type=str, help='optim method, Adam or SGD')
        parser.add_argument('--eval_metric', default='accuracy', type=str, help='clustering eval metirc:mean_accu, accuracy')
        parser.add_argument('--cdd_loss_weight', default=10, type=float, help='weight of cdd loss')
        parser.add_argument('--max_iter_pretrain', default=100, type=int, help='max iter per CDD loop')
        parser.add_argument('--max_iter_train', default=100, type=int, help='max iter per CDD loop')

        args = parser.parse_args()

        args.device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')
        args.in_channels = int((args.time_win*args.sample_freq)/args.down_sample)
        args.save_dir = os.path.join('./DDGCNN_cross_sub_GMM_no_varitional/save_model/', sub)
        self.args = args
        self._set_seed(self.args.seed)

        # self.args.writer = SummaryWriter(log_dir=self.args.save_dir + '/log/', comment='comment',
        #                                  filename_suffix="_test_your_filename_suffix")
        # loss
        self.args.epoch = 0
        self.args.step = -1

        self._print_args()

    def get_args(self):
        return self.args

    def _print_args(self):
        self.logger.info("==========       args      =============")
        for arg, content in self.args.__dict__.items():
            self.logger.info("{}:{}".format(arg, content))
        self.logger.info("==========     args END    =============")
        self.logger.info("\n")



    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False



