import torch
import torch.nn as nn
from DDGCNN_cross_sub_GMM_3sub.solver_utils import to_cuda, to_onehot, filter_samples, filter_class, split_samples_classwise,Sample,get_centers
from DDGCNN_cross_sub_GMM_3sub.cdd import CDD
from DDGCNN_cross_sub_GMM_3sub.clustering import Clustering
from DDGCNN_cross_sub_GMM_3sub.base_solver import BaseSolver
import DDGCNN_cross_sub_GMM_3sub.mmd_utils as mmd
from collections import defaultdict
from utils_cross_sub.loss import discrepancy, msda_regulizer


class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy, self).__init__()

    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y * torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss

def distribution_loss(source_sigma,source_mu,target_sigma,target_mu):
    dis_loss = 0.
    for i in range(len(source_mu)):
        dis_loss += (target_sigma-source_sigma[i])+(torch.exp(source_sigma[i]**2)+(source_mu[i]-target_mu)**2)/(2*torch.exp(target_sigma)**2)-0.5
    dis_loss /= len(source_sigma)
    return dis_loss

class CANSolver(BaseSolver):
    def __init__(self, extractor, classifier, record_file, top_coff, dataloader, opt, clustering_source_name, rest_sub_name, clustering_target_name=None, logger=None, num_layers=1,
                 kernel_num=(5,5), kernel_mul=(2,2),num_classes=4,intra_only=False,device='cuda',save_dir=None,**kwargs):
        super(CANSolver, self).__init__(extractor, classifier, record_file, top_coff, dataloader, opt, clustering_source_name, rest_sub_name, clustering_target_name=None, logger=None, num_layers=1,
                 kernel_num=(5,5), kernel_mul=(2,2),num_classes=4,intra_only=False,device='cuda',save_dir=None,**kwargs)
        self.cdd = CDD(num_layers=num_layers,kernel_num=kernel_num,kernel_mul=kernel_mul,num_classes=num_classes,intra_only=intra_only)
        self.clustering = Clustering(opt.clustering_eps, dist_type=opt.dist_type)
        self.device=device
        self.num_classes = num_classes
        self.entropy_minimization_loss = one_hot_CrossEntropy()
        self.logger = logger
        self.save_dir = save_dir

    # def complete_training(self):
    #     if self.loop >= self.opt.max_loop:
    #         self.record_file.write('target_acc'+' '+str(self.best_acc) +' '+
    #                                str(self.best_source_acc[0])+' '+
    #                                str(self.best_source_acc[1])+' '+
    #                                str(self.best_source_acc[2])+' '+
    #                                str(self.best_source_acc[3])+' '+
    #                                '\n')
    #         return True
    #     if 'target_centers' not in self.history or \
    #             'ts_center_dist' not in self.history or \
    #             'target_labels' not in self.history:
    #         return False
    #     if len(self.history['target_centers']) < 2 or \
    #             len(self.history['ts_center_dist']) < 1 or \
    #             len(self.history['target_labels']) < 2:
    #         return False
    #     target_centers = self.history['target_centers']
    #     eval1 = 0
    #     for i in range(len(self.clustering_source_name)):
    #         eval1 += torch.mean(self.clustering.Dist.get_dist(target_centers[-1][i],
    #                                                      target_centers[-2][i])).item()
    #     eval1/=len(self.clustering_source_name)
    #     eval2 = self.history['ts_center_dist'][-1].item()
    #     path2label_hist = self.history['target_labels']
    #     paths = self.clustered_target_samples['data']
    #     num = 0
    #     for path in paths:
    #         pre_label = path2label_hist[-2][path]
    #         cur_label = path2label_hist[-1][path]
    #         if pre_label != cur_label:
    #             num += 1
    #     eval3 = 1.0 * num / len(paths)
    #     return (eval1 < self.opt.stop_thresholds[0] and eval2 < self.opt.stop_thresholds[1] and eval3 < self.opt.stop_thresholds[2])

    def solve(self):
        self.best_acc = 0
        self.best_rest_mae = 0
        self.best_rest_sem = 0
        self.best_source_acc=[0]*len(self.clustering_source_name)
        self.train_net()
        self.record_file.write('target_acc' + ' ' + str(self.best_acc) + ' ' +
                               str(self.best_source_acc[0]) + ' ' +
                               str(self.best_source_acc[1]) + ' ' +
                               str(self.best_source_acc[2]) + ' ' +
                               str(self.best_source_acc[3]) + ' ' +
                               '\n')
        # while True:
        #     with torch.no_grad():
        #         self.update_labels()
        #         self.clustered_target_samples = self.clustering.samples
        #         target_centers = self.clustering.centers
        #         center_change = self.clustering.center_change
        #         path2label = self.clustering.path2label
        #
        #         self.register_history('target_centers', target_centers, 2)
        #         self.register_history('ts_center_dist', center_change, 2)
        #         self.register_history('target_labels', path2label, 2)
        #
        #         if self.clustered_target_samples is not None and \
        #                 self.clustered_target_samples['gt'] is not None:
        #             preds = to_onehot(self.clustered_target_samples['label'],
        #                               self.opt.class_num)
        #             gts = self.clustered_target_samples['gt']
        #             res = self.model_eval(preds, gts)
        #             self.target_hypt, self.filtered_classes = self.filtering()
        #             print('Clustering %s: %.4f' % (self.opt.eval_metric, res))
        #         stop = self.complete_training()
        #         if stop: break
        #
        #     self.update_network()
        #     self.loop += 1
        #     print('loop:',self.loop)
        print('Training Done!')

    # def update_labels(self):
    #     self.extractor.eval()
    #     self.classifier.eval()
    #     source_centers = []
    #     for source_name in range(len(self.clustering_source_name)):
    #         source_dataloader = self.train_data[source_name]
    #         source_centers.append(get_centers(self.extractor,source_dataloader, self.opt.class_num, self.device))
    #
    #     init_target_centers = source_centers
    #     target_dataloader = self.train_data[len(self.clustering_source_name)]
    #     self.clustering.set_init_centers(init_target_centers)
    #     self.clustering.feature_clustering(self.extractor, target_dataloader, device=self.device)

    # def filtering(self):
    #     threshold = self.opt.filtering_threshold
    #     min_sn_cls = self.opt.min_sn_sample
    #     target_samples = self.clustered_target_samples
    #
    #     chosen_samples = filter_samples(
    #         target_samples, threshold=threshold)
    #
    #     filtered_classes = filter_class(
    #         chosen_samples['label'], min_sn_cls, self.opt.class_num)
    #
    #     print('The number of filtered classes: %d.' % len(filtered_classes))
    #     return chosen_samples, filtered_classes
    #
    # def construct_categorical_dataloader(self, samples, filtered_classes):
    #     target_classwise = split_samples_classwise(
    #         samples, self.opt.class_num)
    #
    #     dataloader = self.train_data[-1]
    #     classnames = ['1','2','3','4']
    #     dataloader.class_set = [classnames[c] for c in filtered_classes]
    #     dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] for c in filtered_classes}
    #     dataloader.target_gt = {classnames[c]: target_classwise[c]['gt'] for c in filtered_classes}
    #     dataloader.construct()
    #     self.cdd_iter = dataloader.whole_iter
    #     self.class_wise_loader = dataloader.dataloader()

    # def update_network(self):
    #     stop = False
    #     update_iters = 0
    #     epoch = -1
    #
    #     while not stop:
    #         epoch += 1
    #         self.update_lr()
    #
    #         self.extractor.train()
    #         self.classifier.train()
    #         self.extractor.zero_grad()
    #         self.classifier.zero_grad()
    #         self.construct_categorical_dataloader(self.target_hypt, self.filtered_classes)
    #
    #         ce_loss_iter = 0
    #         source_kl_iter = 0
    #         target_kl_iter = 0
    #         cdd_loss_iter = 0
    #         source_accuracy = [0.] * len(self.clustering_source_name)
    #         target_accuracy = 0.
    #         mu_bias_loss_iter = 0.
    #         sigma_bias_loss_iter = 0.
    #         source_total_num = 0.
    #         target_total_num = 0.
    #         for j in range(self.cdd_iter):
    #             source_data, tar_batch = next(self.class_wise_loader)
    #             num_s_count=[int(source_data[0][0].shape[0]/self.num_classes)]*self.num_classes
    #             source_total_num += source_data[0][0].shape[0]
    #             target_total_num += tar_batch[0].shape[0]
    #             num_t_count = num_s_count
    #             num_s_count = [num_s_count]*len(self.clustering_source_name)
    #
    #             mu_bias_loss_o = 0.
    #             sigma_bias_loss_o = 0.
    #             ce_loss_o = 0.
    #             kl_loss_o = 0.
    #             gen_source_sample_list = []
    #             source_sigma_list = []
    #             source_mu_list = []
    #             for i in range(len(self.clustering_source_name)):
    #                 mu, sigma, mu_bias, sigma_bias, source_sample, bias_sample = self.extractor(source_data[i][0],
    #                                                                                             source_data[i][1],
    #                                                                                             source_data[i][2],
    #                                                                                             source_data[i][3])
    #                 source_preds = self.classifier(source_sample)
    #                 gen_source_sample_list.append(source_sample)
    #                 source_sigma_list.append(sigma+sigma_bias)
    #                 source_mu_list.append(mu+mu_bias)
    #
    #                 ce_loss = 10 * self.CELoss(source_preds, source_data[i][4].long())
    #                 source_KL_divergence = 0.5 * ((mu ** 2 + sigma.exp() - 1 - sigma) / 2).mean()
    #                 mu_bias_loss = 0.0003 * self.ReconLoss(mu_bias, torch.ones_like(mu_bias) * 1e-7)
    #                 sigma_bias_loss = 0.0003 * self.ReconLoss(sigma, torch.ones_like(sigma_bias) * 1e-7)
    #                 loss = ce_loss + source_KL_divergence + mu_bias_loss + sigma_bias_loss
    #                 loss.backward(retain_graph=True)
    #
    #                 mu_bias_loss_o += mu_bias_loss
    #                 sigma_bias_loss_o += sigma_bias_loss
    #                 ce_loss_o += ce_loss
    #                 kl_loss_o += source_KL_divergence
    #
    #                 ce_loss_iter += ce_loss
    #                 mu_bias_loss_iter += mu_bias_loss
    #                 sigma_bias_loss_iter += sigma_bias_loss
    #                 source_kl_iter += source_KL_divergence
    #                 source_accuracy[i] += torch.sum(torch.argmax(source_preds, dim=1) == source_data[i][4]).item()
    #             if len(self.filtered_classes) > 0:
    #                 mu_target, sigma_target, mu_bias, sigma_bias, gen_target_sample, gen_target_sample_bias = self.extractor(
    #                     tar_batch[0], tar_batch[1], tar_batch[2], tar_batch[3])
    #                 target_preds = self.classifier(gen_target_sample_bias)
    #                 cdd_loss = self.cdd.forward(gen_source_sample_list, gen_target_sample_bias, num_s_count, num_t_count)['cdd']
    #                 target_KL_divergence = 0.5 * ((mu_target ** 2 + sigma_target.exp() - 1 - sigma_target) / 2).mean()
    #                 cdd_loss *= self.opt.cdd_loss_weight
    #                 loss = cdd_loss + target_KL_divergence
    #                 loss.backward(retain_graph=True)
    #                 cdd_loss_iter += cdd_loss
    #                 target_kl_iter += target_KL_divergence
    #                 target_accuracy += torch.sum(torch.argmax(target_preds, dim=1) == tar_batch[4]).item()
    #
    #             self.extractor_optimizer.step()
    #             self.classifier_optimizer.step()
    #             self.extractor_optimizer.zero_grad()
    #             self.classifier_optimizer.zero_grad()
    #         source_accuracy_ = [source_accuracy[t]/source_total_num*100. for t in range(len(source_accuracy))]
    #         target_accuracy_ = target_accuracy/target_total_num*100
    #         self.source_test_accuracy, target_test_accuracy = self.test()
    #
    #         self.logger.info('Epoch :{}\t CE_loss :{:.3f}\t CDD_loss :{:.3f}\t Source KL :{:.3f}\t Target KL :{:.3f}\t Mu_bias :{:.3f}\t Sigma_bias :{:.3f}\t Target train acc :{:.2f}\t Target test_acc :{:.2f}\t'.format(
    #             epoch, ce_loss_iter/4, cdd_loss_iter, source_kl_iter,  target_kl_iter, mu_bias_loss_iter/4, sigma_bias_loss_iter/4, target_accuracy_, target_test_accuracy))
    #         for i in range(len(self.clustering_source_name)):
    #             self.logger.info('Epoch :{}\t Source {} train acc :{:.2f}\t test_acc :{:.2f}'.format(epoch, self.clustering_source_name[i], source_accuracy_[i],self.source_test_accuracy[i]))
    #
    #         if target_test_accuracy >= self.best_acc:
    #             self.best_acc = target_test_accuracy
    #             self.best_source_acc = self.source_test_accuracy
    #             filename = '{}/model.pth'.format(self.save_dir)
    #             torch.save({'loop': self.loop,
    #                         'iters': self.iters,
    #                         'extractor_state_dict': self.extractor.state_dict(),
    #                         'classifier_state_dict': self.classifier.state_dict(),
    #                         'extractor_optimizer_state_dict': self.extractor_optimizer.state_dict(),
    #                         'classifier_optimizer_state_dict': self.classifier_optimizer.state_dict(),
    #                         }, filename)
    #             print('Model saved.')
    #         print('Best target test acc:', self.best_acc)
    #
    #         update_iters += 1
    #         self.iters += 1
    #         if update_iters >= self.iters_train:
    #             stop = True
    #         else:
    #             stop = False

    def train_net(self):
        stop = False
        update_iters = 0
        epoch = -1

        while not stop:
            epoch += 1
            # self.update_lr()
            #
            # self.extractor.train()
            # self.classifier.train()
            # self.extractor.zero_grad()
            # self.classifier.zero_grad()
            source_loss = {}
            source_loss['Extractor_loss'] = 0.
            source_loss['Step1_loss'] = 0.
            source_loss['Step2_loss'] = 0.
            source_loss['Step3_loss'] = 0.
            source_loss['Target valid Acc'] = 0.
            source_loss['Target train Acc'] = 0.
            for source in self.clustering_source_name:
                source_loss[source] = {}
                source_loss[source]['Predictor_loss'] = 0.
                source_loss[source]['Source valid Acc'] = 0.
                source_loss[source]['Target valid Acc'] = 0.
                source_loss[source]['Target train Acc'] = 0.
                for i in range(1, 3):  # C1 C2 loss
                    source_loss[source]['C' + str(i) + '_loss'] = 0.
                    source_loss[source]['C' + str(i) + '_Acc'] = 0.
                    source_loss[source]['Source valid C' + str(i) + '_Acc'] = 0.
                    source_loss[source]['Target valid C' + str(i) + '_Acc'] = 0.
                    source_loss[source]['Target valid C' + str(i) + '_loss'] = 0.
                    source_loss[source]['Target train C' + str(i) + '_Acc'] = 0.
                    source_loss[source]['Target train C' + str(i) + '_loss'] = 0.

            self.extractor.train()
            source_ac = {}
            for source in self.clustering_source_name:
                self.classifier[source]['c1'].train()
                self.classifier[source]['c2'].train()
                source_ac[source] = defaultdict(int)

            ce_loss_iter = 0
            source_kl_iter = 0
            target_kl_iter = 0
            cdd_loss_iter = 0
            source_accuracy = [0.]*len(self.clustering_source_name)
            target_accuracy = 0.
            source_total_num = 0
            target_total_num = 0
            target_num = 0.

            Step1_loss = 0.
            Step2_loss = 0.
            Step3_loss = 0.
            kl_loss_coff = 1e-4
            for i, (src_batch,tar_batch) in enumerate(zip(zip(*self.train_data[:len(self.clustering_source_name)]),self.train_data[len(self.clustering_source_name)])):
                gen_source_sample_list = []
                loss_cls = 0
                for index, data in enumerate(src_batch):
                    data1, data2, data3, data4, y, path = data[0], data[1], data[2], data[3], data[4], data[5]
                    data1, data2, data3, data4, y = data1.to(self.device), data2.to(self.device), data3.to(self.device), data4.to(self.device), y.to(self.device)

                    mu, sigma, source_sample = self.extractor(data1, data2, data3, data4)
                    # source_preds = self.classifier(source_sample)

                    pred1 = self.classifier[self.clustering_source_name[index]]['c1'](source_sample)
                    pred2 = self.classifier[self.clustering_source_name[index]]['c2'](source_sample)
                    gen_source_sample_list.append(source_sample)

                    # ce_loss = 10*self.CELoss(source_preds, y.long())

                    source_loss[self.clustering_source_name[index]]['C1_Acc'] += torch.sum(
                        torch.argmax(pred1, dim=1) == y).item()
                    source_loss[self.clustering_source_name[index]]['C2_Acc'] += torch.sum(
                        torch.argmax(pred2, dim=1) == y).item()
                    loss1 = self.CELoss(pred1, y.long())
                    source_loss[self.clustering_source_name[index]]['C1_loss'] += loss1
                    loss2 = self.CELoss(pred2, y.long())
                    source_loss[self.clustering_source_name[index]]['C2_loss'] += loss2
                    loss_cls += (loss1 + loss2)
                    ce_loss_iter += loss_cls/2
                    source_loss[self.clustering_source_name[index]]['Predictor_loss'] += loss_cls

                    source_KL_divergence = kl_loss_coff*torch.mean(0.5 * torch.sum(((mu**2 + sigma.exp() - 1 - sigma) / 2),1),0)
                    # loss = ce_loss + source_KL_divergence
                    loss_cls += source_KL_divergence
                    # loss.backward(retain_graph=True)

                    # ce_loss_o += ce_loss

                    # ce_loss_iter += ce_loss
                    source_kl_iter += source_KL_divergence

                    # source_accuracy[index] += torch.sum(torch.argmax(source_preds, dim=1) == y).item()
                    # labels_onehot = to_onehot(y.int(), self.num_classes)
                    # count = torch.sum(labels_onehot, dim=0).type(torch.cuda.IntTensor)
                    # num_s_count.append(count)

                source_total_num += data1.shape[0]
                data1, data2, data3, data4, y_t = tar_batch[0], tar_batch[1], tar_batch[2], tar_batch[3], tar_batch[4]
                mu_target, sigma_target, gen_target_sample = self.extractor(
                    data1, data2, data3, data4)

                target_num += data1.shape[0]
                # target_preds = self.classifier(gen_target_sample)
                sigma = [0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15]
                cdd_loss = 0
                for m in range(len(self.clustering_source_name)):
                    for n in range(m, len(self.clustering_source_name)):
                        cdd_loss += mmd.mix_rbf_mmd2(gen_source_sample_list[m], gen_source_sample_list[n], sigma)
                    cdd_loss += mmd.mix_rbf_mmd2(gen_source_sample_list[m], gen_target_sample, sigma)
                cdd_loss *= self.opt.cdd_loss_weight
                cdd_loss_iter += cdd_loss

                target_KL_divergence = kl_loss_coff*torch.mean(0.5 * torch.sum(((mu_target**2 + sigma_target.exp() - 1 - sigma_target) / 2),1),0)

                target_kl_iter += target_KL_divergence
                loss_1 = target_KL_divergence+cdd_loss+loss_cls
                loss_1.backward(retain_graph=True)
                self.extractor_optimizer.zero_grad()
                for source in self.clustering_source_name:
                    self.classifier_optimizer[source]['c1'].zero_grad()
                    self.classifier_optimizer[source]['c2'].zero_grad()
                loss_1.backward()
                self.extractor_optimizer.step()
                for source in self.clustering_source_name:
                    self.classifier_optimizer[source]['c1'].step()
                    self.classifier_optimizer[source]['c2'].step()
                    self.classifier_optimizer[source]['c1'].zero_grad()
                    self.classifier_optimizer[source]['c2'].zero_grad()
                self.extractor_optimizer.zero_grad()

                mu_target, sigma_target, gen_target_sample = self.extractor(data1, data2, data3, data4)
                target_KL_divergence = kl_loss_coff*torch.mean(0.5 * torch.sum(((mu_target**2 + sigma_target.exp() - 1 - sigma_target) / 2),1),0)
                d_loss = 0
                c_loss = target_KL_divergence
                for index, data in enumerate(src_batch):
                    data1, data2, data3, data4, y = data[0], data[1], data[2], data[3], data[4]
                    # dd_loss = 0
                    cc_loss = 0
                    data1, data2, data3, data4, y = data1.to(self.opt.device), data2.to(self.opt.device), data3.to(
                        self.opt.device), data4.to(self.opt.device), y.to(self.opt.device)
                    mu, sigma, source_sample = self.extractor(data1, data2, data3, data4)
                    source_KL_divergence = kl_loss_coff*torch.mean(0.5 * torch.sum(((mu**2 + sigma.exp() - 1 - sigma) / 2),1),0)
                    pred1 = self.classifier[self.clustering_source_name[index]]['c1'](source_sample)
                    pred2 = self.classifier[self.clustering_source_name[index]]['c2'](source_sample)

                    loss1 = self.CELoss(pred1, y.long())
                    loss2 = self.CELoss(pred2, y.long())
                    cc_loss += (loss1 + loss2)
                    c_loss += cc_loss+source_KL_divergence

                    pred_c1 = self.classifier[self.clustering_source_name[index]]['c1'](gen_target_sample)
                    pred_c2 = self.classifier[self.clustering_source_name[index]]['c2'](gen_target_sample)
                    combine1 = (torch.softmax(pred_c1, dim=1) + torch.softmax(pred_c2, dim=1)) / 2
                    dd_loss = discrepancy(pred_c1, pred_c2)
                    d_loss += dd_loss

                    for index_2, o_batch in enumerate(src_batch[index + 1:]):
                        pred_2_c1 = self.classifier[self.clustering_source_name[index_2 + index]]['c1'](gen_target_sample)
                        pred_2_c2 = self.classifier[self.clustering_source_name[index_2 + index]]['c2'](gen_target_sample)
                        combine2 = (torch.softmax(pred_2_c1, dim=1) + torch.softmax(pred_2_c2, dim=1)) / 2
                        dd_loss = discrepancy(combine1, combine2) * 0.1
                        d_loss += dd_loss
                    source_loss[self.clustering_source_name[index]]['Predictor_loss'] += (cc_loss - dd_loss)
                loss_2 = c_loss - d_loss
                Step2_loss += loss_2
                loss_2.backward()
                self.extractor_optimizer.zero_grad()
                for source in self.clustering_source_name:
                    self.classifier_optimizer[source]['c1'].zero_grad()
                    self.classifier_optimizer[source]['c2'].zero_grad()
                for source in self.clustering_source_name:
                    self.classifier_optimizer[source]['c1'].step()
                    self.classifier_optimizer[source]['c2'].step()
                    self.classifier_optimizer[source]['c1'].zero_grad()
                    self.classifier_optimizer[source]['c2'].zero_grad()

                # cdd_loss_iter += cdd_loss
                # target_kl_iter += target_KL_divergence
                # target_total_num += data1.shape[0]
                # target_accuracy += torch.sum(torch.argmax(target_preds, dim=1) == y).item()
                # self.extractor_optimizer.step()
                # self.classifier_optimizer.step()
                # self.extractor_optimizer.zero_grad()
                # self.classifier_optimizer.zero_grad()
                "Use target classify discrepancy loss update Extractor"
                for k in range(4):
                    data = tar_batch
                    data1, data2, data3, data4, y = data[0], data[1], data[2], data[3], data[4]
                    mu_target, sigma_target, gen_target_sample = self.extractor(data1, data2, data3, data4)
                    target_KL_divergence = kl_loss_coff*torch.mean(0.5 * torch.sum(((mu_target**2 + sigma_target.exp() - 1 - sigma_target) / 2),1),0)
                    discrepency_loss = target_KL_divergence
                    for index in range(len(self.clustering_source_name)):

                        pred_c1 = self.classifier[self.clustering_source_name[index]]['c1'](gen_target_sample)
                        pred_c2 = self.classifier[self.clustering_source_name[index]]['c2'](gen_target_sample)
                        combine1 = (torch.softmax(pred_c1, dim=1) + torch.softmax(pred_c2, dim=1)) / 2
                        discrepency_loss += discrepancy(pred_c1, pred_c2)

                        for index_2, o_batch in enumerate(src_batch[index + 1:]):
                            pred_2_c1 = self.classifier[self.clustering_source_name[index_2 + index]]['c1'](gen_target_sample)
                            pred_2_c2 = self.classifier[self.clustering_source_name[index_2 + index]]['c2'](gen_target_sample)
                            combine2 = (torch.softmax(pred_2_c1, dim=1) + torch.softmax(pred_2_c2, dim=1)) / 2
                            discrepency_loss += discrepancy(combine1, combine2) * 0.1
                    Step3_loss += discrepency_loss
                    self.extractor.zero_grad()
                    for source in self.clustering_source_name:
                        self.classifier_optimizer[source]['c1'].zero_grad()
                        self.classifier_optimizer[source]['c2'].zero_grad()
                    discrepency_loss.backward()
                    self.extractor_optimizer.step()
                    self.extractor_optimizer.zero_grad()
                    for source in self.clustering_source_name:
                        self.classifier_optimizer[source]['c1'].zero_grad()
                        self.classifier_optimizer[source]['c2'].zero_grad()

                source_loss['Extractor_loss'] += (Step1_loss + Step3_loss)
                source_loss['Step1_loss'] += Step1_loss
                source_loss['Step2_loss'] += Step2_loss
                source_loss['Step3_loss'] += Step3_loss

                final_pred = 1
                for index_s in range(len(self.clustering_source_name)):
                    pred1 = self.classifier[self.clustering_source_name[index_s]]['c1'](gen_target_sample)
                    pred2 = self.classifier[self.clustering_source_name[index_s]]['c2'](gen_target_sample)
                    source_loss[self.clustering_source_name[index_s]]['Target train C1_loss'] += self.CELoss(pred1,
                                                                                                             y_t.long())
                    source_loss[self.clustering_source_name[index_s]]['Target train C2_loss'] += self.CELoss(pred2,
                                                                                                             y_t.long())
                    source_loss[self.clustering_source_name[index_s]]['Target train C1_Acc'] += torch.sum(
                        torch.argmax(pred1, dim=1) == y_t).item()
                    source_loss[self.clustering_source_name[index_s]]['Target train C2_Acc'] += torch.sum(
                        torch.argmax(pred2, dim=1) == y_t).item()
                    full_pred = (torch.softmax(pred1, dim=1) + torch.softmax(pred2, dim=1)) / 2
                    source_loss[self.clustering_source_name[index_s]]['Target train Acc'] += torch.sum(
                        torch.argmax(full_pred, dim=1) == y_t).item()

                    if isinstance(final_pred, int):
                        final_pred = (torch.softmax(pred1, dim=1) + torch.softmax(pred2, dim=1)) / 2
                    else:
                        final_pred += (torch.softmax(pred1, dim=1) + torch.softmax(pred2, dim=1)) / 2
                source_loss['Target train Acc'] += torch.sum(torch.argmax(final_pred, dim=1) == y_t).item()
                self.extractor_optimizer.zero_grad()
                for source in self.clustering_source_name:
                    self.classifier_optimizer[source]['c1'].zero_grad()
                    self.classifier_optimizer[source]['c2'].zero_grad()

            source_loss['Target train Acc'] = source_loss['Target train Acc'] / target_num * 100.
            for source in self.classifier_optimizer:
                source_loss[source]['Target train C1_Acc'] = source_loss[source][
                                                                 'Target train C1_Acc'] / target_num * 100.
                source_loss[source]['Target train C2_Acc'] = source_loss[source][
                                                                 'Target train C2_Acc'] / target_num * 100.
                source_loss[source]['Target train Acc'] = source_loss[source]['Target train Acc'] / target_num * 100.
                source_loss[source]['Target train C1_loss'] /= (i + 1)
                source_loss[source]['Target train C2_loss'] /= (i + 1)

            source_loss['Extractor_loss'] /= i + 1
            source_loss['Step1_loss'] /= i + 1
            source_loss['Step2_loss'] /= i + 1
            source_loss['Step3_loss'] /= i + 1
            for index, source in enumerate(self.clustering_source_name):
                source_loss[source]['Predictor_loss'] /= i
                for k in range(1, 3):
                    source_loss[source]['C' + str(k) + '_loss'] /= i
                    source_loss[source]['C' + str(k) + '_Acc'] = source_loss[source]['C' + str(k) + '_Acc'] / \
                                                                 source_total_num * 100
            # source_accuracy_ = [source_accuracy[t] / source_total_num * 100. for t in range(len(source_accuracy))]
            # target_accuracy_ = target_accuracy / target_total_num * 100
            # self.source_test_accuracy, target_test_accuracy = self.test()
            source_loss = self.MSDA_test(source_loss)
            self.logger.info(
                'Epoch :{}\t CE_loss :{:.3f}\t CDD_loss :{:.3f}\t Source KL :{:.3f}\t Target KL :{:.3f}\t Target train acc :{:.2f}\t '
                'Target test_acc :{:.2f}\t'.format(epoch, ce_loss_iter / len(self.clustering_source_name), cdd_loss_iter, source_kl_iter, target_kl_iter,
                                                   source_loss['Target train Acc'], source_loss['Target valid Acc']))
            for i in range(len(self.clustering_source_name)):
                self.logger.info('Epoch :{}\t Source {} train C1 acc :{:.2f}\t train C2 acc :{:.2f}\t test_C1 acc :{:.2f}\t test_C2 acc :{:.2f}'.format(epoch,
                                                                                                    self.clustering_source_name[i],
                                                                                                    source_loss[self.clustering_source_name[i]]['C1_Acc'],
                                                                                                    source_loss[self.clustering_source_name[i]]['C2_Acc'],
                                                                                                    source_loss[self.clustering_source_name[i]]['Source valid C1_Acc'],
                                                                                                    source_loss[self.clustering_source_name[i]]['Source valid C2_Acc'],))

            if source_loss['Target valid Acc'] >= self.best_acc:
                self.best_acc = source_loss['Target valid Acc']
                # self.best_source_acc = self.source_test_accuracy
                for source_indes in range(len(self.clustering_source_name)):
                    self.best_source_acc[source_indes] = source_loss[self.clustering_source_name[source_indes]]['Source valid Acc']

                filename = '{}/model.pth'.format(self.save_dir)
                # torch.save({'loop': self.loop,
                #             'iters': self.iters,
                #             'extractor_state_dict': self.extractor.state_dict(),
                #             'classifier_state_dict': self.classifier.state_dict(),
                #             'extractor_optimizer_state_dict': self.extractor_optimizer.state_dict(),
                #             'classifier_optimizer_state_dict': self.classifier_optimizer.state_dict(),
                #             }, filename)
                # print('Model saved.')
            print('Best target test acc:', self.best_acc)
            update_iters += 1
            self.iters += 1
            if update_iters >= self.opt.max_iter_pretrain:
                stop = True
            else:
                stop = False

    def MSDA_test(self,source_loss):
        self.extractor.eval()
        for source in self.clustering_source_name:
            self.classifier[source]['c1'].eval()
            self.classifier[source]['c2'].eval()
        with torch.no_grad():
            target_num = 0.
            source_num = [0.]*len(self.clustering_source_name)
            for i, (src_batch,tar_batch) in enumerate(zip(zip(*self.test_data[:len(self.clustering_source_name)]),self.test_data[len(self.clustering_source_name)])):
                data = tar_batch
                data1, data2, data3, data4, y_t = data[0], data[1], data[2], data[3], data[4]
                mu_target, sigma_target, gen_target_sample = self.extractor(data1, data2, data3, data4)
                target_num += data1.shape[0]
                final_pred = 1
                for index_s, data in enumerate(src_batch):
                    data1, data2, data3, data4, y_s = data[0], data[1], data[2], data[3], data[4]
                    mu, sigma, source_sample = self.extractor(data1, data2, data3, data4)
                    pred1 = self.classifier[self.clustering_source_name[index_s]]['c1'](source_sample)
                    pred2 = self.classifier[self.clustering_source_name[index_s]]['c2'](source_sample)
                    source_loss[self.clustering_source_name[index_s]]['Source valid C1_Acc'] += torch.sum(
                        torch.argmax(pred1, dim=1) == y_s).item()
                    source_loss[self.clustering_source_name[index_s]]['Source valid C2_Acc'] += torch.sum(
                        torch.argmax(pred2, dim=1) == y_s).item()
                    full_pred = (torch.softmax(pred1, dim=1) + torch.softmax(pred2, dim=1)) / 2
                    source_loss[self.clustering_source_name[index_s]]['Source valid Acc'] += torch.sum(
                        torch.argmax(full_pred, dim=1) == y_s).item()
                    source_num[index_s]+=data1.shape[0]

                    pred1 = self.classifier[self.clustering_source_name[index_s]]['c1'](gen_target_sample)
                    pred2 = self.classifier[self.clustering_source_name[index_s]]['c2'](gen_target_sample)
                    source_loss[self.clustering_source_name[index_s]]['Target valid C1_loss'] += self.CELoss(pred1, y_t.long())
                    source_loss[self.clustering_source_name[index_s]]['Target valid C2_loss'] += self.CELoss(pred2, y_t.long())
                    source_loss[self.clustering_source_name[index_s]]['Target valid C1_Acc'] += torch.sum(
                        torch.argmax(pred1, dim=1) == y_t).item()
                    source_loss[self.clustering_source_name[index_s]]['Target valid C2_Acc'] += torch.sum(
                        torch.argmax(pred2, dim=1) == y_t).item()
                    full_pred = (torch.softmax(pred1, dim=1) + torch.softmax(pred2, dim=1)) / 2
                    source_loss[self.clustering_source_name[index_s]]['Target valid Acc'] += torch.sum(
                        torch.argmax(full_pred, dim=1) == y_t).item()
                    if isinstance(final_pred, int):
                        final_pred = (torch.softmax(pred1, dim=1) + torch.softmax(pred2, dim=1)) / 2
                    else:
                        final_pred += (torch.softmax(pred1, dim=1) + torch.softmax(pred2, dim=1)) / 2
                source_loss['Target valid Acc'] += torch.sum(torch.argmax(final_pred, dim=1) == y_t).item()

            source_loss['Target valid Acc'] = source_loss['Target valid Acc'] / target_num * 100.
            for index_s,source in enumerate(self.clustering_source_name):
                source_loss[source]['Source valid C1_Acc'] = source_loss[source][
                                                                 'Source valid C1_Acc'] / source_num[index_s] * 100.
                source_loss[source]['Source valid C2_Acc'] = source_loss[source][
                                                                 'Source valid C2_Acc'] / source_num[index_s] * 100.
                source_loss[source]['Source valid Acc'] = source_loss[source]['Source valid Acc'] / source_num[index_s] * 100.
                source_loss[source]['Target valid C1_Acc'] = source_loss[source][
                                                                 'Target valid C1_Acc'] / target_num * 100.
                source_loss[source]['Target valid C2_Acc'] = source_loss[source][
                                                                 'Target valid C2_Acc'] / target_num * 100.
                source_loss[source]['Target valid Acc'] = source_loss[source]['Target valid Acc'] / target_num * 100.
                source_loss[source]['Target valid C1_loss'] /= (i + 1)
                source_loss[source]['Target valid C2_loss'] /= (i + 1)

        # self.logger.info('Target valid Acc{:.2f}'.format(source_loss['Target valid Acc']))
        # for source in self.clustering_source_name:
        #     self.logger.info(
        #         'Source :{},Target valid C1 Acc :{:.2f},Target valid C2 Acc :{:.2f},Target valid Acc :{:.2f}, Source valid C1 Acc :{:.2f},Source valid C2 Acc :{:.2f},Source valid Acc :{:.2f}'.format(
        #             source,
        #             source_loss[source]['Target valid C1_Acc'], source_loss[source]['Target valid C2_Acc'],
        #             source_loss[source]['Target valid Acc'],
        #             source_loss[source]['Source valid C1_Acc'], source_loss[source]['Source valid C2_Acc'],
        #             source_loss[source]['Source valid Acc']))
        return source_loss

