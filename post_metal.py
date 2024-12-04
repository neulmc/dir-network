import numpy as np
import cv2
import os
import shutil
import matplotlib.pyplot as plt

pred_dir1 = 'E:/real-LMCuncertain/new-LMC/RAW_inst/100'
pred_dir2 = 'E:/real-LMCuncertain/new-LMC/Ensemble_DE_inst/100'
pred_dir3 = 'E:/real-LMCuncertain/new-LMC/Ensemble_BS_inst/100'
pred_dir4 = 'E:/real-LMCuncertain/new-LMC/BBB_gauss_inst/100'
pred_dir5 = 'E:/real-LMCuncertain/new-LMC/BBB_laplace_inst/100'
pred_dir6 = 'E:/real-LMCuncertain/new-LMC/MCDrop_inst/100'
pred_dir7 = 'E:/real-LMCuncertain/new-LMC/Beta_inst_w1e-3_r2(sel)/100'

gt_dir = 'E:/real-LMCuncertain/dataset/preheat3/split_images/gt'
task = 'inst'

values = [64, 128, 255]
gt_values = [1, 2, 3]
keys = dict()
keys[64] = 'c2'
keys[128] = 'c1'
keys[255] = 'c3'
gt_keys = dict()
gt_keys[1] = 'c1'
gt_keys[2] = 'c2'
gt_keys[3] = 'c3'
#temps = ['15-30','45-60','90-120']
#temps = ['15','30','45','60','90','120']
temps = ['15','30-45','60-90','120']
min_area = 10
uncer_t = 0.4
uncer_ratio = 0.4

def build_lst():
    files = []
    with open("E:/real-LMCuncertain/dataset/preheat3/" + task + "_test.lst", "r") as f:
        for line in f.readlines():
            files.append(line.split('\n')[0].split(' ')[1].split('gt/')[1])  # 去掉列表中每一个元素的换行符
    return files

def inst_growing(img, values, keys, uncertain, uncertain_lab,
                 c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes, c1_uncer,
                 c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes, c2_uncer,
                 c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes, c3_uncer):
    class Point(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def getX(self):
            return self.x

        def getY(self):
            return self.y
    def getGrayDiff(img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))
    def selectConnects():
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]  # 八邻域
        return connects
    def regionGrow(img, seeds, seedMark, label, thresh=1):
        height, weight = img.shape
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        connects = selectConnects()
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)  # 弹出第一个元素
            seedMark[currentPoint.x, currentPoint.y] = label
            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = label
                    seedList.append(Point(tmpX, tmpY))
        return seedMark
    def getmetric(seedmark):
        num_ = 0
        h_ = []
        w_ = []
        size_ = []
        uncer_ = []
        for i in range(int(np.max(seedmark))):
            if i == 0 or len(np.where(seedmark == i)[0]) == 0:
                continue
            num_ += 1
            xs, ys = np.where(seedmark == i)
            size_.append(len(xs))
            xs_len = np.max(xs) - np.min(xs) + 1
            ys_len = np.max(ys) - np.min(ys) + 1
            if xs_len > ys_len:
                h_.append(xs_len)
                w_.append(ys_len)
            else:
                h_.append(ys_len)
                w_.append(xs_len)
            if uncertain_lab:
                sz = uncertain[seedmark == i]
                #num_uncer_ = np.sqrt(np.sum(sz > uncer_t))
                num_uncer_ = np.sum(sz > uncer_t)
                uncer_.append(num_uncer_)
            else:
                uncer_.append(0)
        h_ = np.array(h_)
        w_ = np.array(w_)
        size_ = np.array(size_)
        if uncertain_lab:
            uncer_ = np.array(uncer_)
        return [num_], h_, w_, size_, uncer_

    for value in values:
        seedMark = np.zeros(img.shape)
        label = 0
        seedFull = np.zeros(img.shape)
        seedFull[np.where(img == value)] = 1
        while(len(np.where(seedMark>0)[0]) != len(np.where(seedFull==1)[0])):
            label += 1
            xs, ys = np.where(np.logical_and(seedMark ==0, seedFull==1) == True)
            seeds = [Point(xs[0],ys[0])]
            seedMark = regionGrow(img, seeds, seedMark, label)
        if keys[value] == 'c1':
            c1_seedMarks.append(seedMark)
            nums_, h_, w_, size_, uncer_ = getmetric(seedMark)
            c1_nums.extend(nums_)
            c1_hs.extend(h_)
            c1_ws.extend(w_)
            c1_sizes.extend(size_)
            c1_uncer.extend(uncer_)
        elif keys[value] == 'c2':
            c2_seedMarks.append(seedMark)
            nums_, h_, w_, size_, uncer_ = getmetric(seedMark)
            c2_nums.extend(nums_)
            c2_hs.extend(h_)
            c2_ws.extend(w_)
            c2_sizes.extend(size_)
            c2_uncer.extend(uncer_)
        elif keys[value] == 'c3':
            c3_seedMarks.append(seedMark)
            nums_, h_, w_, size_, uncer_ = getmetric(seedMark)
            c3_nums.extend(nums_)
            c3_hs.extend(h_)
            c3_ws.extend(w_)
            c3_sizes.extend(size_)
            c3_uncer.extend(uncer_)

    return c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes, c1_uncer,\
        c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes, c2_uncer,\
        c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes, c3_uncer

def eval(dirname, values, keys):
    print(dirname)
    files = build_lst()
    for temp in temps:
        c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes, c1_uncer  = [], [], [], [], [], []
        c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes, c2_uncer = [], [], [], [], [], []
        c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes, c3_uncer = [], [], [], [], [], []
        for file in files:
            if file.split('_')[0] in temp:
                img = cv2.imread(dirname + '/' + file, 0)
                if img.shape[1] == 1128:
                    img = img[:,752:]
                    uncertain = cv2.imread(dirname + '/' + file.replace('.png', '_UEM(NormEn).png'), 0)[:,:376] / 255
                    uncertain_lab = True
                else:
                    uncertain = None
                    uncertain_lab = False
                c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes, c1_uncer, \
                c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes, c2_uncer, \
                c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes, c3_uncer = inst_growing(img, values, keys, uncertain, uncertain_lab,
                                                                      c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes, c1_uncer,
                                                                      c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes, c2_uncer,
                                                                      c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes, c3_uncer,)
        c1_nums, c1_hs, c1_ws, c1_sizes, c1_uncer = np.array(c1_nums), np.array(c1_hs), np.array(c1_ws), np.array(c1_sizes), np.array(c1_uncer)
        c2_nums, c2_hs, c2_ws, c2_sizes, c2_uncer = np.array(c2_nums), np.array(c2_hs), np.array(c2_ws), np.array(c2_sizes), np.array(c2_uncer)
        c3_nums, c3_hs, c3_ws, c3_sizes, c3_uncer = np.array(c3_nums), np.array(c3_hs), np.array(c3_ws), np.array(c3_sizes), np.array(c3_uncer)

        c1_idx = np.where(c1_sizes < min_area)
        c2_idx = np.where(c2_sizes < min_area)
        c3_idx = np.where(c3_sizes < min_area)

        c1_hs, c1_ws, c1_sizes, c1_uncer = np.delete(c1_hs, c1_idx), np.delete(c1_ws, c1_idx), np.delete(c1_sizes, c1_idx), np.delete(c1_uncer, c1_idx)
        c2_hs, c2_ws, c2_sizes, c2_uncer = np.delete(c2_hs, c2_idx), np.delete(c2_ws, c2_idx), np.delete(c2_sizes, c2_idx), np.delete(c2_uncer, c2_idx)
        c3_hs, c3_ws, c3_sizes, c3_uncer = np.delete(c3_hs, c3_idx), np.delete(c3_ws, c3_idx), np.delete(c3_sizes, c3_idx), np.delete(c3_uncer, c3_idx)

        #c1_idx_uncer = np.where((c1_uncer) / (c1_sizes) > uncer_ratio)
        c1_idx_uncer = np.where(np.logical_and((c1_uncer - (c1_hs + c1_ws) * 2) / (c1_sizes - (c1_hs + c1_ws) * 2) > uncer_ratio, c1_uncer > (c1_hs + c1_ws) * 2))
        c2_idx_uncer = np.where(np.logical_and((c2_uncer - (c2_hs + c2_ws) * 2) / (c2_sizes - (c2_hs + c2_ws) * 2) > uncer_ratio, c2_uncer > (c2_hs + c2_ws) * 2))
        c3_idx_uncer = np.where(np.logical_and((c3_uncer - (c3_hs + c3_ws) * 2) / (c3_sizes - (c3_hs + c3_ws) * 2) > uncer_ratio, c3_uncer > (c3_hs + c3_ws) * 2))

        c1_hs, c1_ws, c1_sizes, c1_uncer = np.delete(c1_hs, c1_idx_uncer), np.delete(c1_ws, c1_idx_uncer), np.delete(c1_sizes, c1_idx_uncer), np.delete(c1_uncer, c1_idx_uncer)
        c2_hs, c2_ws, c2_sizes, c2_uncer = np.delete(c2_hs, c2_idx_uncer), np.delete(c2_ws, c2_idx_uncer), np.delete(c2_sizes, c2_idx_uncer), np.delete(c2_uncer, c2_idx_uncer)
        c3_hs, c3_ws, c3_sizes, c3_uncer = np.delete(c3_hs, c3_idx_uncer), np.delete(c3_ws, c3_idx_uncer), np.delete(c3_sizes, c3_idx_uncer), np.delete(c3_uncer, c3_idx_uncer)

        temp_c1_nums, temp_c2_nums, temp_c3_nums = np.mean(c1_nums)-(len(c1_idx[0])+len(c1_idx_uncer[0]))/len(c1_nums), np.mean(c2_nums)-len(c2_idx)/len(c2_nums), np.mean(c3_nums)-len(c3_idx)/len(c3_nums)
        temp_c1_lens, temp_c2_lens, temp_c3_lens = np.mean(np.sqrt(c1_hs**2+c1_ws**2)), np.mean(np.sqrt(c2_hs**2+c2_ws**2)), np.mean(np.sqrt(c3_hs**2+c3_ws**2))
        temp_c1_uncern_lens, temp_c2_uncern_lens, temp_c3_uncern_lens = np.mean(c1_uncer), np.mean(c2_uncer), np.mean(c3_uncer)
        temp_c1_size, temp_c2_size, temp_c3_size = np.mean(c1_sizes), np.mean(c2_sizes), np.mean(c3_sizes)
        temp_c1_radius, temp_c2_radius, temp_c3_radius = np.sqrt(temp_c1_size/np.pi), np.sqrt(temp_c2_size/np.pi), np.sqrt(temp_c3_size/np.pi)
        temp_c1_radius_uncern, temp_c2_radius_uncern, temp_c3_radius_uncern = np.sqrt(temp_c1_uncern_lens/np.pi), np.sqrt(temp_c2_uncern_lens/np.pi), np.sqrt(temp_c3_uncern_lens/np.pi)
        temp_c1_hwratio, temp_c2_hwratio, temp_c3_hwratio = np.mean(c1_hs/c1_ws), np.mean(c2_hs/c2_ws), np.mean(c3_hs/c3_ws)

        print('[temp]:' + temp)
        print('[class]:' + 'c1' + ' [nums]: {:.3f}'.format(np.mean(temp_c1_nums)) + ' [lens]: {:.3f}'.format(np.mean(temp_c1_lens)) + ' [radius]: {:.3f}'.format(
            np.mean(temp_c1_radius)) + ' [radius_uncern]: {:.3f}'.format(
            np.mean(temp_c1_radius_uncern)) + ' [radius_uncern_real]: {:.3f}'.format(
            np.mean(temp_c1_radius_uncern) - 2) +' [sizes]: {:.3f}'.format(np.mean(temp_c1_size)) + ' [ratio]: {:.3f}'.format(np.mean(temp_c1_hwratio)))
        '''
        print('[class]:' + 'c2' + ' [nums]: {:.2f}'.format(np.mean(temp_c2_nums)) + ' [lens]: {:.2f}'.format(np.mean(temp_c2_lens)) + ' [radius]: {:.2f}'.format(
            np.mean(temp_c2_radius)) + ' [radius_uncern]: {:.2f}'.format(
            np.mean(temp_c2_radius_uncern)) + ' [sizes]: {:.2f}'.format(np.mean(temp_c2_size)))
        print('[class]:' + 'c3' + ' [nums]: {:.2f}'.format(np.mean(temp_c3_nums)) + ' [lens]: {:.2f}'.format(np.mean(temp_c3_lens)) + ' [radius]: {:.2f}'.format(
            np.mean(temp_c3_radius)) + ' [radius_uncern]: {:.2f}'.format(
            np.mean(temp_c3_radius_uncern)) + ' [sizes]: {:.2f}'.format(np.mean(temp_c3_size)))
        '''
        #print('[class]:' + 'c1' + ' [nums]: {:.2f}'.format(np.mean(temp_c1_nums)) + ' [lens]: {:.2f}'.format(np.mean(temp_c1_lens)) + ' [uncer]: {:.2f}'.format(np.mean(temp_c1_uncern_lens)) + ' [sizes]: {:.2f}'.format(np.mean(temp_c1_size)))
        #print('[class]:' + 'c2' + ' [nums]: {:.2f}'.format(np.mean(temp_c2_nums)) + ' [lens]: {:.2f}'.format(np.mean(temp_c2_lens)) + ' [uncer]: {:.2f}'.format(np.mean(temp_c2_uncern_lens)) + ' [sizes]: {:.2f}'.format(np.mean(temp_c2_size)))
        #print('[class]:' + 'c3' + ' [nums]: {:.2f}'.format(np.mean(temp_c3_nums)) + ' [lens]: {:.2f}'.format(np.mean(temp_c3_lens)) + ' [uncer]: {:.2f}'.format(np.mean(temp_c3_uncern_lens)) + ' [sizes]: {:.2f}'.format(np.mean(temp_c3_size)))



if __name__ == '__main__':
    #eval(gt_dir, gt_values, gt_keys)
    #eval(pred_dir1, values, keys)
    #eval(pred_dir2, values, keys)
    #eval(pred_dir3, values, keys)
    #eval(pred_dir4, values, keys)
    #eval(pred_dir5, values, keys)
    #eval(pred_dir6, values, keys)
    eval(pred_dir7, values, keys)

