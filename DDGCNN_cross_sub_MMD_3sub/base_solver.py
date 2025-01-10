import torch
import torch.nn as nn
import os
from DDGCNN_cross_sub_MMD_3sub.solver_utils import mean_accuracy, accuracy, adjust_learning_rate_exp, adjust_learning_rate_inv, adjust_learning_rate_plateau
from torch import optim
import numpy as np
import math


class BaseSolver:
    def __init__(self, extractor, classifier, record_file, top_coff, dataloader, opt, clustering_source_name, rest_sub_name, clustering_target_name=None, logger=None, num_layers=1,
                 kernel_num=(5,5), kernel_mul=(2,2),num_classes=4, intra_only=False,device='cuda',save_dir=None,**kwargs):
        self.opt = opt
        self.clustering_source_name = clustering_source_name
        self.source_coff = [top_coff[i]/sum(top_coff) for i in range(len(top_coff))]
        self.clustering_target_name = clustering_target_name
        self.record_file = record_file
        self.rest_name = rest_sub_name
        self.extractor = extractor
        self.classifier = classifier
        self.init_data(dataloader)

        self.CELoss = nn.CrossEntropyLoss()
        self.ReconLoss = nn.MSELoss(size_average=False)
        if torch.cuda.is_available():
            self.CELoss.cuda()

        self.loop = 0
        self.iters = 0
        self.iters_train = self.opt.max_iter_train
        self.iters_pretrain = self.opt.max_iter_pretrain
        self.device = device
        self.history = {}

        self.base_lr = self.opt.lr
        self.momentum = self.opt.momentum

        self.build_optimizer()

    def init_data(self, dataloader):
        self.train_data = []
        self.test_data = []
        self.rest_data = []
        for key in dataloader.keys():
            if 'test' in key:
                self.test_data.append(dataloader[key])
            elif 'rest' in key:
                self.rest_data.append(dataloader[key])
            else:
                self.train_data.append(dataloader[key])

    def get_samples_categorical(self, data_name, category):
        assert (data_name in self.train_data)
        assert ('loader' in self.train_data[data_name] and 'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader'][category]
        data_iterator = self.train_data[data_name]['iterator'][category]
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'][category] = data_iterator
        return sample
    # def build_optimizer(self):
    #     assert self.opt.optim in ["Adam", "SGD"], \
    #         "Currently do not support your specified optimizer."
    #     if self.opt.optim == "Adam":
    #         self.extractor_optimizer = optim.Adam(self.extractor.parameters(),
    #                                     lr=self.base_lr,
    #                                     weight_decay=self.opt.weight_decay)
    #         self.classifier_optimizer = optim.Adam(self.classifier.parameters(),
    #                        lr=self.base_lr,
    #                        weight_decay=self.opt.weight_decay)
    #     elif self.opt.optim == "SGD":
    #         self.extractor_optimizer = optim.SGD(self.extractor.parameters(),
    #                                    lr=self.base_lr, momentum=self.momentum,
    #                                    weight_decay=self.opt.weight_decay)
    #         self.classifier_optimizer=optim.SGD(self.classifier.parameters(),
    #                                                   lr=self.base_lr, momentum=self.momentum,
    #                                                   weight_decay=self.opt.weight_decay)
    # def update_lr(self):
    #     iters = self.iters
    #     if self.opt.lr_schedule == 'exp':
    #         adjust_learning_rate_exp(self.base_lr,self.extractor_optimizer, iters,decay_rate=self.opt.lr_decay_rate,decay_step=self.opt.decay_step)
    #         adjust_learning_rate_exp(self.base_lr,self.classifier_optimizer, iters,decay_rate=self.opt.lr_decay_rate,decay_step=self.opt.decay_step)
    #
    #     elif self.opt.lr_schedule == 'inv':
    #         adjust_learning_rate_inv(self.base_lr, self.extractor_optimizer, iters, self.opt.inv_alpha, self.opt.inv_beta)
    #         adjust_learning_rate_inv(self.base_lr, self.classifier_optimizer, iters, self.opt.inv_alpha, self.opt.inv_beta)
    #     elif self.opt.lr_schedule == 'plateau':
    #         adjust_learning_rate_plateau(self.extractor_optimizer,self.opt.lr_decay_rate, self.opt.optim_patience)
    #         adjust_learning_rate_plateau(self.classifier_optimizer,self.opt.lr_decay_rate, self.opt.optim_patience)
    #     else:
    #         raise NotImplementedError("Currently don't support the specified learning rate schedule: %s." % self.opt.lr_schedule)
    def build_optimizer(self):
        assert self.opt.optim in ["Adam", "SGD"], \
            "Currently do not support your specified optimizer."
        if self.opt.optim == "Adam":
            self.extractor_optimizer = optim.Adam(self.extractor.parameters(),
                                        lr=self.base_lr,
                                        weight_decay=self.opt.weight_decay)
            self.classifier_optimizer = {}
            for i in range(len(self.clustering_source_name)):
                self.classifier_optimizer[self.clustering_source_name[i]]={}
                self.classifier_optimizer[self.clustering_source_name[i]]['c1']=optim.Adam(self.classifier[self.clustering_source_name[i]]['c1'].parameters(),
                           lr=self.base_lr,
                           weight_decay=self.opt.weight_decay)
                self.classifier_optimizer[self.clustering_source_name[i]]['c2']=optim.Adam(self.classifier[self.clustering_source_name[i]]['c2'].parameters(),
                               lr=self.base_lr,
                               weight_decay=self.opt.weight_decay)
        elif self.opt.optim == "SGD":
            self.extractor_optimizer = optim.SGD(self.extractor.parameters(),
                                       lr=self.base_lr, momentum=self.momentum,
                                       weight_decay=self.opt.weight_decay)
            self.classifier_optimizer = {}
            for i in range(len(self.clustering_source_name)):
                self.classifier_optimizer[self.clustering_source_name[i]]={}
                self.classifier_optimizer[self.clustering_source_name[i]]['c1']=optim.SGD(self.classifier[self.clustering_source_name[i]]['c1'].parameters(),
                                                      lr=self.base_lr, momentum=self.momentum,
                                                      weight_decay=self.opt.weight_decay)
                self.classifier_optimizer[self.clustering_source_name[i]]['c2']=optim.SGD(self.classifier[self.clustering_source_name[i]]['c2'].parameters(),
                                                      lr=self.base_lr, momentum=self.momentum,
                                                      weight_decay=self.opt.weight_decay)

    def update_lr(self):
        iters = self.iters
        if self.opt.lr_schedule == 'exp':
            adjust_learning_rate_exp(self.base_lr,self.extractor_optimizer, iters,decay_rate=self.opt.lr_decay_rate,decay_step=self.opt.decay_step)
            for i in range(len(self.clustering_source_name)):
                adjust_learning_rate_exp(self.base_lr,self.classifier_optimizer[self.clustering_source_name[i]]['c1'], iters,decay_rate=self.opt.lr_decay_rate,decay_step=self.opt.decay_step)
                adjust_learning_rate_exp(self.base_lr, self.classifier_optimizer[self.clustering_source_name[i]]['c2'], iters,
                                 decay_rate=self.opt.lr_decay_rate, decay_step=self.opt.decay_step)

        elif self.opt.lr_schedule == 'inv':
            adjust_learning_rate_inv(self.base_lr, self.extractor_optimizer, iters, self.opt.inv_alpha, self.opt.inv_beta)
            for i in range(len(self.clustering_source_name)):
                 adjust_learning_rate_inv(self.base_lr, self.classifier_optimizer[self.clustering_source_name[i]]['c1'], iters, self.opt.inv_alpha, self.opt.inv_beta)
                 adjust_learning_rate_inv(self.base_lr, self.classifier_optimizer[self.clustering_source_name[i]]['c2'],
                                     iters, self.opt.inv_alpha, self.opt.inv_beta)
        elif self.opt.lr_schedule == 'plateau':
            adjust_learning_rate_plateau(self.extractor_optimizer,self.opt.lr_decay_rate, self.opt.optim_patience)
            for i in range(len(self.clustering_source_name)):
                 adjust_learning_rate_plateau(self.classifier_optimizer[self.clustering_source_name[i]]['c1'],self.opt.lr_decay_rate, self.opt.optim_patience)
                 adjust_learning_rate_plateau(self.classifier_optimizer[self.clustering_source_name[i]]['c2'],self.opt.lr_decay_rate, self.opt.optim_patience)
        else:
            raise NotImplementedError("Currently don't support the specified learning rate schedule: %s." % self.opt.lr_schedule)


    def model_eval(self, preds, gts):
        assert (self.opt.eval_metric in ['mean_accu', 'accuracy']), \
            "Currently don't support the evaluation metric you specified."

        if self.opt.eval_metric == "mean_accu":
            res = mean_accuracy(preds, gts)
        else:
            res = accuracy(preds, gts)
        return res

    def save_ckpt(self):
        save_path = self.opt.SAVE_DIR
        ckpt_resume = os.path.join(save_path)
        torch.save({'loop': self.loop,
                    'iters': self.iters,
                    'extractor_state_dict': self.extractor.module.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict(),
                    'extractor_optimizer_state_dict': self.extractor_optimizer.state_dict(),
                    'classifier_optimizer_state_dict': self.classifier_optimizer.state_dict()
                    }, ckpt_resume)

    def complete_training(self):
        if self.loop > self.opt.TRAIN.MAX_LOOP:
            return True

    def register_history(self, key, value, history_len):
        if key not in self.history:
            self.history[key] = [value]
        else:
            self.history[key] += [value]

        if len(self.history[key]) > history_len:
            self.history[key] = \
                self.history[key][len(self.history[key]) - history_len:]

    def test(self):
        self.extractor.eval()
        self.classifier.eval()
        source_accuracy = [0.]*len(self.clustering_source_name)
        target_accuracy = 0.
        source_total_num = 0
        target_total_num = 0
        for i, (src_batch,tar_batch) in enumerate(zip(zip(*self.test_data[:len(self.clustering_source_name)]),self.test_data[len(self.clustering_source_name)])):
            for index, data in enumerate(src_batch):
                    data1, data2, data3, data4, y, path = data[0], data[1], data[2], data[3], data[4], data[5]
                    data1, data2, data3, data4, y = data1.to(self.device), data2.to(self.device), data3.to(self.device), data4.to(
                        self.device), y.to(self.device)

                    feature = self.extractor(data1, data2, data3, data4)
                    source_preds = self.classifier(feature)
                    source_accuracy[index] += torch.sum(torch.argmax(source_preds, dim=1) == y).item()

            source_total_num += data1.shape[0]
            data1, data2, data3, data4, y = tar_batch[0], tar_batch[1], tar_batch[2], tar_batch[3], tar_batch[4]
            feature = self.extractor(data1, data2, data3, data4)
            target_preds = self.classifier(feature)
            target_accuracy += torch.sum(torch.argmax(target_preds, dim=1) == y).item()
            target_total_num += data1.shape[0]
        source_accuracy_ = [acc/source_total_num*100. for acc in source_accuracy]
        target_accuracy_ = target_accuracy/target_total_num*100
        return source_accuracy_, target_accuracy_

    def rest_test(self):
        self.extractor.eval()
        self.classifier.eval()
        rest_total_num = 0
        rest_accuracy = [0.] * len(self.rest_data)
        for i, rest_batch in enumerate(self.rest_data):
            for index, data in enumerate(rest_batch):
                    data1, data2, data3, data4, y, path = data[0], data[1], data[2], data[3], data[4], data[5]
                    data1, data2, data3, data4, y = data1.to(self.device), data2.to(self.device), data3.to(self.device), data4.to(
                        self.device), y.to(self.device)

                    feature = self.extractor(data1, data2, data3, data4)
                    rest_preds = self.classifier(feature)
                    rest_accuracy[index] += torch.sum(torch.argmax(rest_preds, dim=1) == y).item()
            rest_total_num += data1.shape[0]
        rest_accuracy_ = [acc / rest_total_num * 100. for acc in rest_accuracy]
        rest_std = np.std(rest_accuracy_,ddof=1)
        rest_sem = rest_std/math.sqrt(len(rest_accuracy_)-1)
        rest_mean = np.mean(rest_accuracy_)
        return rest_mean, rest_sem


    def solve(self):
        pass

    def update_network(self, **kwargs):
        pass

