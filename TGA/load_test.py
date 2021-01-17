import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from bicubic import imresize
from Guassian import Guassian_downsample

def load_img(image_path, nFrame, scale, num):
    tt = int(nFrame / 2)
    target = modcrop(Image.open(image_path).convert('RGB'),scale) #[W,H]
    char_len = len(image_path)
    HR=[]
    seq = [x for x in range(-tt,tt+1)]
    pad_first = [3,2,1]
    pad_last = [-1,-2,-3]
    count = 0
    for i in seq:
       if i != 0:
           index = int(image_path[char_len-7:char_len-4]) + i
           image = image_path[0:char_len-7]+'{0:03d}'.format(index)+'.png'
           if os.path.exists(image):
               temp = modcrop(Image.open(image).convert('RGB'), scale)
               HR.append(temp)
           elif index <= 0:
               print('{} is not found, using reference lr frame to pad'.format(image))
               index_first = 1
               image_first = image_path[0:char_len-7]+'{0:03d}'.format(index_first)+'.png'
               temp_first = modcrop(Image.open(image_first).convert('RGB'), scale)
               HR.append(temp_first)
               count += 1
           elif index > 0:
               print('{} is not found, using reference lr frame to pad'.format(image))
               index_last = num
               image_last = image_path[0:char_len-7]+'{0:03d}'.format(index_last)+'.png'
               temp_last = modcrop(Image.open(image_last).convert('RGB'), scale)
               HR.append(temp_last)
               count += 1
       else:
           HR.append(target)
    return HR

def modcrop(img,scale):
    (iw, ih) = img.size
    ih = ih - (ih % scale)
    iw = iw - (iw % scale)
    img = img.crop((0,0,iw,ih))
    return img

class DataloadFromFolderTest(data.Dataset): # load test dataset
    def __init__(self, image_dir, file_list, scale, nFrame, transform):
        super(DataloadFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join(image_dir,file_list))] # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
        self.image_filenames = [os.path.join(image_dir,x) for x in alist] # get image path list
        self.scale = scale
        self.transform = transform # To_tensor
        self.nFrame = nFrame
    def __getitem__(self, index):
        GT = load_img(self.image_filenames[index], self.nFrame, self.scale, len(self.image_filenames)) 
        GT = [np.asarray(HR) for HR in GT] 
        GT = np.asarray(GT)
        if self.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale), (0,0)), mode='reflect')
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        GT = GT.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w)
        target = GT[:,3,:,:]
        LR = Guassian_downsample(GT, self.scale)
        HR = imresize(LR[:,3,:,:], 4,  antialiasing=True) 
        LR_new = []
        group1 = torch.stack((LR[:,0,:,:],LR[:,3,:,:],LR[:,-1,:,:]),1)
        group2 = torch.stack((LR[:,2,:,:],LR[:,3,:,:],LR[:,-3,:,:]),1)
        group3 = torch.stack((LR[:,1,:,:],LR[:,3,:,:],LR[:,-2,:,:]),1)
        LR_new = torch.cat((group1,group2,group3),1) 
        return LR_new, target, HR

        
    def __len__(self):
        return len(self.image_filenames) # total video number. not image number

