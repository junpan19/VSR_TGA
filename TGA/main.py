from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set 
import pdb
from torch.optim import lr_scheduler 
import socket
import time
from tensorboardX import SummaryWriter
import cv2
import math
import sys
from utils import Logger 
import numpy as np
from arch import TGA
import datetime
import torchvision.utils as vutils
import random

parser = argparse.ArgumentParser(description='PyTorch TGA Example')
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots. This is a savepoint, using to save training model.')
parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--data_dir', type=str, default='/home/liguangrui3/video/workspace_panj/vimo_x5/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt', help='where record all of image name in dataset.')
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='TGA', help='network name')
parser.add_argument('--stepsize', type=int, default=10, help='Learning rate is decayed by a factor of 10 every half of total epochs')
parser.add_argument('--gamma', type=float, default=0.1 , help='learning rate decay')
parser.add_argument('--save_model_path', type=str, default='./result/weight', help='Location to save checkpoint models')
parser.add_argument('--save_train_log', type=str ,default='./result/log/')
parser.add_argument('--output', type=str , default='./Results', help='Location to save checkpoint models')
parser.add_argument('--weight-decay', default=5e-04, type=float,help="weight decay (default: 5e-04)")
parser.add_argument('--log_name', type=str, default='TGA')
parser.add_argument('--train_data_path', type=str, default='/cache/data')
parser.add_argument('--gpu-devices', default='0,1,2,3,4,5,6,7', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES') 

opt = parser.parse_args()
opt.data_dir = '/home/ma-user/work/data/vimeo_HR_fulldata/sequences'

def main():
    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
    R = opt.scale # upscale factor 
    #if opt.eval_mode:
    #    sys.stdout = Logger(os.path.join(opt.save_test_log,'test_'+systime+'.txt'))
    #else:
    sys.stdout = Logger(os.path.join(opt.save_train_log, 'train_'+opt.log_name+'.txt'))
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else: 
       use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    pin_memory = True if use_gpu else False 

    print(opt)
    print('===> Loading Datasets')
    train_set = get_training_set(opt.data_dir, R, opt.data_augmentation, opt.file_list) 
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchsize, shuffle=True, pin_memory=pin_memory, drop_last=True)
    print('===> DataLoading Finished')
    tga = TGA(R) # initial filter generate network 
    print('===> {} model has been initialized'.format(opt.model_name))
    tga = torch.nn.DataParallel(tga)
    criterion = nn.L1Loss(reduction='sum')
    criterion_discriminate = nn.L1Loss() # Huber loss

    if use_gpu:
        criterion_discriminate = criterion_discriminate.cuda()
        tga = tga.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(tga.parameters(), lr = opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay) 

    if opt.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size = opt.stepsize, gamma=opt.gamma)

    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        train(train_loader, tga, R, criterion, optimizer, epoch, use_gpu, criterion_discriminate) #fed data into network
        scheduler.step() 
        if (epoch) % (opt.snapshots) == 0: 
            checkpoint(tga, epoch)

def train(train_loader, tga, scale, criterion, optimizer, epoch, use_gpu, criterion_discriminate):
    train_mode = True
    epoch_loss = 0
    tga.train()
    for iteration, data in enumerate(train_loader):
        input,target,HR= data[0],data[1],data[2] # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]
        if use_gpu:
            input = Variable(input).cuda()
            target = Variable(target).cuda() 
            HR = Variable(HR).cuda()
        t0 = time.time()
        optimizer.zero_grad()
        prediction = tga(input, HR) 
        loss = criterion(prediction, target)/opt.batchsize
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        t1 = time.time()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(train_loader), loss.item(), (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_loader)))

def test():
    cudnn.benchmark = False
    model.eval()
     
def checkpoint(tga, epoch): 
    save_model_path = os.path.join(opt.save_model_path, systime)
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name  = 'X'+str(opt.scale)+'_{}L'.format(opt.layer)+'_{}'.format(opt.patch_size)+'_epoch_{}.pth'.format(epoch)
    torch.save(tga.state_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path,model_name)))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    main()    
