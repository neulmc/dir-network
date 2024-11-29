from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2
import random

class preheat_Loader(data.Dataset):

    def __init__(self, root='../dataset/preheat', split='train', task='random', shuffle = False):
        self.dir = root
        data_file = root + '/' + task + '_' + split +'.lst'
        self.filelist = open(data_file, 'r').readlines()
        if shuffle:
            random.shuffle(self.filelist)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file, gt_file = self.filelist[index].replace('\n','').split(' ')
        img = np.array(cv2.imread(self.dir + '/' + img_file), dtype=np.float32)
        lb = np.array(cv2.imread(self.dir + '/' + gt_file, 0), dtype=np.int)
        #lb[lb == 255] = 1
        img = np.transpose(img, (2, 0, 1))
        return img, lb, img_file.split('.')[0].split('/')[-1]