import os
from os.path import join
import unet
import cv2
from data_loader import preheat_Loader
from torch.utils.data import DataLoader, sampler
from utils import Logger, Averagvalue, save_checkpoint
import time
import sys
from priors import *
import metric
import shutil
import random
import numpy as np
import math
from torch import lgamma, digamma

mode = 'dir'  # RAW; MCDrop; NaiveDrop; BBB; Ensemble; dir
en_mode = 'DE' # 'DE':'Deep Ensemble' 'BS':'Bootstrapped'
prior_mode = 'guass'  # guass laplace
task = 'inst'  # inst inst_1 inst_2 inst_4 inst_5
name = '_'
TMP_DIR = mode + '_' + task + name
lr = 1e-3
batch_size = 4
maxepoch = 101
eval_save_freq = 100
eval_freq = 20
dataset_dir = 'dataset/preheat3'
print_freq = 10
#MC_iter = 100  # prediction 100
MC_iter = 10
MC_p = 0.5
MC_pB = 0.1
BBB_sample = 3  # train
dir_l1_num = 2000
dir_l1_weight = 0.001
if prior_mode == 'guass':
    prior_instance = isotropic_gauss_prior(mu=0, sigma=0.1)  # BBB_gauss
else:
    prior_instance = laplace_prior(mu=0, b=0.1)  # BBB_laplace
model_c = 32  # model channel
mc_pq = 0  # mc prior
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1029
en_Num = 10
mappings = dict()
mappings[1] = 128
mappings[2] = 64
mappings[3] = 255

if isinstance(prior_instance, isotropic_gauss_prior) :
    TMP_DIR = TMP_DIR.replace('BBB','BBB_gauss')
else:
    TMP_DIR = TMP_DIR.replace('BBB','BBB_laplace')
TMP_DIR = TMP_DIR.replace('Ensemble','Ensemble_' + en_mode)

def seed_torch(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

def cross_entropy_loss(output, label, eps=1e-8):
    #label = torch.unsqueeze(label, dim=1)
    one_hot_labels = torch.nn.functional.one_hot(label.long(), num_classes=4)
    one_hot_labels = one_hot_labels.permute(0, 3, 1, 2)
    #return - torch.sum(label * torch.log(torch.clamp(output, min=eps)) +
    #                   (1 - label) * torch.log(torch.clamp(1 - output, min=eps)))
    return - torch.sum(one_hot_labels * torch.log(torch.clamp(output, min=eps)))

def dir_expected_loglikelihood(output, label, eps=1e-8):
    one_hot_labels = torch.nn.functional.one_hot(label.long(), num_classes=4)
    one_hot_labels = one_hot_labels.permute(0, 3, 1, 2)
    dia_sum = digamma(torch.clamp(torch.sum(output, dim=1, keepdim=True), min=eps))
    dia_id = digamma(torch.clamp(output, min=eps))
    cost = one_hot_labels * (dia_sum - dia_id)
    return torch.sum(cost)

def dir_l1(output):
    cost = torch.mean(torch.abs(dir_l1_num - torch.clamp(torch.sum(output, dim = 1), min = dir_l1_num)))* dir_l1_weight  # l1
    return torch.sum(cost)

def train(model, train_loader, optimizer, epoch, batch_size):
    print('Traing Ep:%d' % epoch)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    losses_l2 = Averagvalue()
    losses_mlp = Averagvalue()
    losses_kl = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    Nbatch = len(train_loader)
    if mode == 'Ensemble':
        model_batch_id = -1 * np.ones(Nbatch)
        if en_mode == 'BS':
            for jk in range(Nbatch):
                model_batch_id[jk] = jk % en_Num
        elif en_mode == 'DE':
            sample_num = math.floor(Nbatch / en_Num)
            for jk in range(en_Num):
                model_batch_id_tmp = np.where(model_batch_id == -1)[0]
                sele_non = np.random.choice(model_batch_id_tmp, sample_num, replace=False)
                model_batch_id[sele_non] = jk
            x_ = np.where(model_batch_id == -1)[0]
            model_batch_id[x_] = np.random.randint(en_Num, size=(x_.shape))

    for i, (image, label, filename) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        _, c, w, h = image.shape
        #print(filename)
        if model.mode == 'Ensemble':

            model_id = int(model_batch_id[i])
            outputs = model(image, model_id)
            loss = cross_entropy_loss(outputs, label) / batch_size
            loss.backward()
            optimizer[model_id].step()
            optimizer[model_id].zero_grad()

            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))
            epoch_loss.append(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                       'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                       'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses)
                print(info)

        elif model.mode == 'BBB':
            mlpdw_cum = 0
            Edkl_cum = 0
            for _ in range(BBB_sample):
                outputs, tlqw, tlpw = model(image, sample=True)
                mlpdw_i = cross_entropy_loss(outputs, label)
                # mlpdw_i = F.cross_entropy(out, y, reduction='sum')
                Edkl_i = (tlqw - tlpw) / Nbatch
                mlpdw_cum = mlpdw_cum + mlpdw_i
                Edkl_cum = Edkl_cum + Edkl_i
            mlpdw = mlpdw_cum / BBB_sample / batch_size
            Edkl = Edkl_cum / BBB_sample / batch_size
            loss = Edkl + mlpdw
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))
            losses_mlp.update(mlpdw.item(), image.size(0))
            losses_kl.update(Edkl.item(), image.size(0))
            epoch_loss.append(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                       'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                       'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                       'Loss_mlp {loss_mlp.val:f} (avg:{loss_mlp.avg:f}) '.format(loss_mlp=losses_mlp) + \
                       'Loss_kl {loss_kl.val:f} (avg:{loss_kl.avg:f}) '.format(loss_kl=losses_kl)
                print(info)

        elif model.mode == 'MCDrop' or model.mode == 'NaiveDrop':
            outputs, tpq = model(image)
            mlpdw = cross_entropy_loss(outputs, label) / batch_size
            pq = tpq * mc_pq
            loss = pq + mlpdw
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))
            losses_mlp.update(mlpdw.item(), image.size(0))
            losses_kl.update(pq.item(), image.size(0))
            epoch_loss.append(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                       'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                       'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                       'Loss_mlp {loss_mlp.val:f} (avg:{loss_mlp.avg:f}) '.format(loss_mlp=losses_mlp) + \
                       'Loss_kl {loss_kl.val:f} (avg:{loss_kl.avg:f}) '.format(loss_kl=losses_kl)
                print(info)

        elif model.mode == 'dir':
            outputs = model(image)
            loss1 = dir_expected_loglikelihood(outputs, label) / batch_size
            loss2 = dir_l1(outputs) / batch_size
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # measure accuracy and record loss
            losses.update(loss1.item(), image.size(0))
            losses_l2.update(loss2.item(), image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                       'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                       'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                       'L2Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_l2)
                print(info)

        else:
            outputs = model(image)
            loss = cross_entropy_loss(outputs, label) / batch_size
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))
            epoch_loss.append(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                       'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                       'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses)
                print(info)

def save(model, epoch):
    # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
            }, filename=join(TMP_DIR, "epoch-%d-checkpoint.pth" % epoch))

def test(model, test_loader, epoch):
    def build_uncertainmap3(pred, gt, prob, mode='NormEn'):
        if mode == 'NormEn':
            uncertainty = (- (prob * np.log(prob + 1e-8)) / np.log(4) * 255).astype(np.uint8)
            uncertainty = np.sum(uncertainty, axis=2).astype(np.uint8)
        elif mode == 'MaxP':
            uncertainty = prob.max(axis=2)
            uncertainty = 1 - uncertainty
            uncertainty = uncertainty #* 4 # 注意 是不是4 不对！
            uncertainty = (uncertainty * 255).astype(np.uint8)
        error = (1 * ~np.equal(pred, gt) * 255).astype(np.uint8)
        uncertainty = cv2.cvtColor(uncertainty, cv2.COLOR_GRAY2RGB)
        error = cv2.cvtColor(error, cv2.COLOR_GRAY2RGB)
        merge = np.zeros(uncertainty.shape)
        merge[:, :, 1] = uncertainty[:, :, 1]
        merge[:, :, 2] = error[:, :, 2]
        UEM = np.concatenate((uncertainty, error, merge), axis=1)
        return UEM

    def build_predictions(outputs, label):
        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]
        prob = outputs[0].permute(1, 2, 0).detach().cpu().numpy()
        pred_show = pred.copy()
        pred_show[pred == 1] = mappings[1]
        pred_show[pred == 2] = mappings[2]
        pred_show[pred == 3] = mappings[3]
        lab = label.detach().cpu().numpy()[0]
        lab_show = lab.copy()
        lab_show[lab == 1] = mappings[1]
        lab_show[lab == 2] = mappings[2]
        lab_show[lab == 3] = mappings[3]
        return pred, prob, pred_show, lab_show, lab

    print('Testing')
    save_dir = TMP_DIR + '/' + str(epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.eval()
    if model.mode == 'RAW':
        for i, (image, label, filename) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            filename = filename[0]
            outputs = model(image)
            image_np = image.detach().cpu().numpy()[0]
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            pred, prob, pred_show, lab_show, lab = build_predictions(outputs, label)
            imggt = np.hstack([image_np, lab_show, pred_show])
            cv2.imwrite(save_dir + '/' + filename + '.png', imggt)
            UEM_NE = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='NormEn')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(NormEn).png', UEM_NE)
            UEM_MP = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='MaxP')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(MaxP).png', UEM_MP)
            np.save(save_dir + '/' + filename + '_prob', prob)
    elif model.mode == 'dir':
        for i, (image, label, filename) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            filename = filename[0]
            outputs = model(image)
            image_np = image.detach().cpu().numpy()[0]
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            outputs_prob = outputs / torch.sum(outputs, dim=1, keepdim=True)
            pred, prob, pred_show, lab_show, lab = build_predictions(outputs_prob, label)
            imggt = np.hstack([image_np, lab_show, pred_show])
            cv2.imwrite(save_dir + '/' + filename + '.png', imggt)
            UEM_NE = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='NormEn')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(NormEn).png', UEM_NE)
            UEM_MP = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='MaxP')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(MaxP).png', UEM_MP)
            np.save(save_dir + '/' + filename + '_prob', prob)
    elif model.mode == 'NaiveDrop':
        for i, (image, label, filename) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            filename = filename[0]
            outputs, _ = model(image)
            image_np = image.detach().cpu().numpy()[0]
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            pred, prob, pred_show, lab_show, lab = build_predictions(outputs, label)
            imggt = np.hstack([image_np, lab_show, pred_show])
            cv2.imwrite(save_dir + '/' + filename + '.png', imggt)
            UEM_NE = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='NormEn')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(NormEn).png', UEM_NE)
            UEM_MP = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='MaxP')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(MaxP).png', UEM_MP)
            np.save(save_dir + '/' + filename + '_prob', prob)
    elif model.mode == 'MCDrop':
        for i, (image, label, filename) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            filename = filename[0]
            image_np = image.detach().cpu().numpy()[0]
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            outputs_mean = torch.zeros(label.shape[0], 4, label.shape[1], label.shape[2]).cuda()
            for j in range(MC_iter):
                outputs, _ = model(image)
                _, _, pred_show, _, _ = build_predictions(outputs, label)
                outputs_mean += outputs
                if epoch % eval_save_freq == 0:
                    cv2.imwrite(save_dir + '/' + filename + '_MC' + str(j) + '.png', pred_show)
            outputs_mean = outputs_mean / MC_iter
            pred, prob, pred_show, lab_show, lab = build_predictions(outputs_mean, label)
            imggt = np.hstack([image_np, lab_show, pred_show])
            cv2.imwrite(save_dir + '/' + filename + '.png', imggt)
            UEM_NE = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='NormEn')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(NormEn).png', UEM_NE)
            UEM_MP = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='MaxP')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(MaxP).png', UEM_MP)
            np.save(save_dir + '/' + filename + '_prob', prob)
    elif model.mode == 'BBB':
        for i, (image, label, filename) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            filename = filename[0]
            image_np = image.detach().cpu().numpy()[0]
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            outputs_mean = torch.zeros(label.shape[0], 4, label.shape[1], label.shape[2]).cuda()
            for j in range(MC_iter):
                outputs, _, _ = model(image)
                _, _, pred_show, _, _ = build_predictions(outputs, label)
                outputs_mean += outputs
                if epoch % eval_save_freq == 0:
                    cv2.imwrite(save_dir + '/' + filename + '_BBB' + str(j) + '.png', pred_show)
            outputs_mean = outputs_mean / MC_iter
            pred, prob, pred_show, lab_show, lab = build_predictions(outputs_mean, label)
            imggt = np.hstack([image_np, lab_show, pred_show])
            cv2.imwrite(save_dir + '/' + filename + '.png', imggt)
            UEM_NE = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='NormEn')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(NormEn).png', UEM_NE)
            UEM_MP = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='MaxP')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(MaxP).png', UEM_MP)
            np.save(save_dir + '/' + filename + '_prob', prob)
    elif model.mode == 'Ensemble':
        for i, (image, label, filename) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            filename = filename[0]
            image_np = image.detach().cpu().numpy()[0]
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            outputs_mean = torch.zeros(label.shape[0], 4, label.shape[1], label.shape[2]).cuda()
            for j in range(en_Num):
                outputs = model(image, j)
                _, _, pred_show, _, _ = build_predictions(outputs, label)
                outputs_mean += outputs
                if epoch % eval_save_freq == 0:
                    cv2.imwrite(save_dir + '/' + filename + '_ENS' + str(j) + '.png', pred_show)
            outputs_mean = outputs_mean / en_Num
            pred, prob, pred_show, lab_show, lab = build_predictions(outputs_mean, label)
            imggt = np.hstack([image_np, lab_show, pred_show])
            cv2.imwrite(save_dir + '/' + filename + '.png', imggt)
            UEM_NE = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='NormEn')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(NormEn).png', UEM_NE)
            UEM_MP = build_uncertainmap3(pred.copy(), lab.copy(), prob.copy(), mode='MaxP')
            cv2.imwrite(save_dir + '/' + filename + '_UEM(MaxP).png', UEM_MP)
            np.save(save_dir + '/' + filename + '_prob', prob)

def eval(test_loader, epoch):
    _eval(test_loader, epoch, mode='NormEn')
    if epoch % eval_save_freq == 0:
        _eval(test_loader, epoch, mode='MaxP')

# EVAL
def _eval(test_loader, epoch, mode='NormEn', suffix=''):
    print('Eval ' + mode)
    load_dir = TMP_DIR + '/' + str(epoch)
    acc = []
    dice1_fenzi = 0
    dice1_fenmu = 0
    dice2_fenzi = 0
    dice2_fenmu = 0
    dice3_fenzi = 0
    dice3_fenmu = 0
    mi = []
    une_dice = []  # uncertainty-error
    cor_dice = []  # confidence-correct
    ece = []
    auroc = []
    aupr = []
    mpu = []
    mtu = []
    mean_error = []
    std_error = []
    kl_dis = []
    mean_error_fix = []
    std_error_fix = []
    kl_dis_fix = []
    for i, (image, label, filename) in enumerate(test_loader):
        filename = filename[0] + suffix + '.png'
        imggt = cv2.imread(load_dir + '/' + filename, 0)
        h, w = imggt.shape
        w_ = w / 3
        gt = imggt[:, int(w_):int(w_ * 2)]
        pred = imggt[:, int(w_ * 2):]
        gt[gt == mappings[1]] = 1
        gt[gt == mappings[2]] = 2
        gt[gt == mappings[3]] = 3
        pred[pred == mappings[1]] = 1
        pred[pred == mappings[2]] = 2
        pred[pred == mappings[3]] = 3
        prob = np.load(load_dir + '/' + filename.replace('.png','_prob.npy'))
        if mode == 'NormEn':
            uncertainty = - (prob * np.log(prob + 1e-8)) / np.log(4)
            uncertainty = np.sum(uncertainty, axis=2)
        elif mode == 'MaxP':
            uncertainty = prob.max(axis=2)
            uncertainty = 1 - uncertainty
            uncertainty = uncertainty #* 4 # 注意 是不是4
        acc_, dice_fenzi, dice_fenmu = metric.compute_accurate(pred.copy(), gt.copy())
        dice1_fenzi += dice_fenzi[0]
        dice2_fenzi += dice_fenzi[1]
        dice3_fenzi += dice_fenzi[2]
        dice1_fenmu += dice_fenmu[0]
        dice2_fenmu += dice_fenmu[1]
        dice3_fenmu += dice_fenmu[2]
        mi_ = metric.compute_mi(pred.copy(), gt.copy(), uncertainty.copy())
        une_dice_, cor_dice_ = metric.compute_overlap(pred.copy(), gt.copy(), uncertainty.copy())
        ece_ = metric.compute_calibration(pred.copy(), gt.copy(), uncertainty.copy(), prob.copy())
        auroc_, aupr_, mpu_, mtu_ = metric.compute_auroc(pred.copy(), gt.copy(), uncertainty.copy())
        #mean_error_, std_error_, kl_dis_ = metric.measurement(pred.copy(), gt.copy(), uncertainty.copy())
        #mean_error_fix_, std_error_fix_, kl_dis_fix_ = metric.measurement(pred.copy(), gt.copy(), uncertainty.copy(), uncertain_fixed=True)
        acc.append(acc_)
        mi.append(mi_)
        une_dice.append(une_dice_)
        cor_dice.append(cor_dice_)
        ece.append(ece_)
        auroc.append(auroc_)
        aupr.append(aupr_)
        mpu.append(mpu_)
        mtu.append(mtu_)
        '''
        mean_error.append(mean_error_)
        std_error.append(std_error_)
        kl_dis.append(kl_dis_)
        mean_error_fix.append(mean_error_fix_)
        std_error_fix.append(std_error_fix_)
        kl_dis_fix.append(kl_dis_fix_)
        '''

    acc = np.mean(np.array(acc))
    dice1 = dice1_fenzi / dice1_fenmu
    dice2 = dice2_fenzi / dice2_fenmu
    dice3 = dice3_fenzi / dice3_fenmu
    mi = np.mean(np.array(mi))
    une_dice = np.mean(np.array(une_dice))
    cor_dice = np.mean(np.array(cor_dice))
    ece = np.mean(np.array(ece))
    auroc = np.mean(np.array(auroc))
    aupr = np.mean(np.array(aupr))
    mpu = np.mean(np.array(mpu))
    mtu = np.mean(np.array(mtu))

    print('acc: ' + str(acc))
    print('dice1: ' + str(dice1))
    print('dice2: ' + str(dice2))
    print('dice3: ' + str(dice3))
    print('mDice: ' + str((dice1 + dice2 + dice3) / 3))
    print('mi: ' + str(mi))
    print('une_dice: ' + str(une_dice))
    print('cor_dice: ' + str(cor_dice))
    print('ece: ' + str(ece))
    print('auroc: ' + str(auroc))
    print('aupr: ' + str(aupr))
    print('mpu: ' + str(mpu))
    print('mtu: ' + str(mtu))
    '''
    print('mean_error: ' + str(mean_error))
    print('std_error: ' + str(std_error))
    print('kl_dis: ' + str(kl_dis))
    print('mean_error_fix: ' + str(mean_error_fix))
    print('std_error_fix: ' + str(std_error_fix))
    print('kl_dis_fix: ' + str(kl_dis_fix))
    '''

if __name__ == '__main__':
    seed_torch()
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    shutil.copy(os.path.abspath(__file__), TMP_DIR + '/config')

    # model & optimation
    if mode == 'Ensemble':
        # data
        train_dataset = preheat_Loader(root=dataset_dir, split="train", task=task, shuffle = True)
        test_dataset = preheat_Loader(root=dataset_dir, split="test", task=task)
        # No shuffer
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            num_workers=1, drop_last=True, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=1,
            num_workers=1, drop_last=False, shuffle=False)

        model = unet.UNets(n_Num = en_Num, model_c=model_c)
        model.cuda()
        optimizer = []
        for model_id in range(en_Num):
            optimizer.append(torch.optim.Adam(params=model.models[model_id].parameters(), lr=lr))
    else:
        # data
        train_dataset = preheat_Loader(root=dataset_dir, split="train", task=task)
        test_dataset = preheat_Loader(root=dataset_dir, split="test", task=task)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            num_workers=1, drop_last=True, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=1,
            num_workers=1, drop_last=False, shuffle=False)

        model = unet.UNet(mode=mode, prior_instance=prior_instance, droprate=MC_p, droprateBack=MC_pB, model_c=model_c)
        model.cuda()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # log
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('adam', lr)))
    sys.stdout = log

    for epoch in range(maxepoch):
        # train
        train(model, train_loader, optimizer, epoch, batch_size)  # train_MCdrop train_BBB
        if epoch % eval_freq == 0 and epoch != 0:
            save(model, epoch) # save model
            test(model, test_loader, epoch)  # test_MCdrop test_BBB
            eval(test_loader, epoch)  # eval
        log.flush()  # write log

