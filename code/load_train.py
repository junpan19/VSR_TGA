import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from bicubic import imresize
from Guassian import Guassian_downsample

def load_img(image_path, scale):
    HR = []
    for img_num in range(7):
        GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
        HR.append(GT_temp)
    return HR

def modcrop(img,scale):
    (ih, iw) = img.size
    ih = ih - ( ih % scale)
    iw = iw - ( iw % scale)
    img = img.crop((0,0,ih,iw))
    return img

def train_process(GT, flip_h=True, rot=True, converse=True): # input:list, target:PIL.Image
    if random.random() < 0.5 and flip_h: 
        GT = [ImageOps.flip(LR) for LR in GT]
    if rot:
        if random.random() < 0.5:
            GT = [ImageOps.mirror(LR) for LR in GT]
    group1 = []
    group2 = []
    group3 = []
    group1.append(GT[0])
    group1.append(GT[3])
    group1.append(GT[-1])
    group2.append(GT[2])
    group2.append(GT[3])
    group2.append(GT[-3])
    group3.append(GT[1])
    group3.append(GT[3])
    group3.append(GT[-2])
    GT = []
    GT.extend(group1)
    GT.extend(group2)
    GT.extend(group3)
    return GT

class DataloadFromFolder(data.Dataset): # load train dataset
    def __init__(self, image_dir, scale, data_augmentation, file_list, transform):
        super(DataloadFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join(image_dir,file_list))] # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
        self.image_filenames = [os.path.join(image_dir,x) for x in alist] # get image path list
        self.scale = scale
        self.transform = transform # To_tensor
        self.data_augmentation = data_augmentation # flip and rotate
    def __getitem__(self, index):
        GT = load_img(self.image_filenames[index], self.scale)
        GT = train_process(GT) # input: list (contain PIL), target: PIL
        GT = [np.asarray(HR) for HR in GT]  # PIL -> numpy # input: list (contatin numpy: [H,W,C])
        GT = np.asarray(GT) # numpy, [T,H,W,C], stack with temporal dimension
        if self.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0,0),(2*self.scale,2*self.scale),(2*self.scale,2*self.scale),(0,0)), mode='reflect')
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        GT = GT.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w) # Tensor, [C,T,H,W]
        reference_GT = GT[:,1,:,:]
        LR = Guassian_downsample(GT, self.scale)
        reference_HR = imresize(LR[:,1,:,:],4,antialiasing=True)
        return LR, reference_GT, reference_HR
        

    def __len__(self):
        return len(self.image_filenames) # total video number. not image number

