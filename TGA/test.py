from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_test_set #, get_eval_set
import time
import torch.backends.cudnn as cudnn
import cv2
import math
import sys
import datetime
from utils import Logger
import numpy as np
import torchvision.utils as vutils
from arch import TGA

parser = argparse.ArgumentParser(description='PyTorch TGA Example')
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testbatchsize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--cuda',default=True, type=bool)
parser.add_argument('--nFrames', default=7, type=int, help='size of temporal radius')
parser.add_argument('--model_name', type=str, default='TGA', help='network name')
parser.add_argument('--test_dir',type=str,default='//home/ma-user/work/data/Vid4')
parser.add_argument('--file_test_list',type=str, default ='',help='where record all of image name in dataset.')
parser.add_argument('--save_test_log', type=str,default='./log/test')
parser.add_argument('--pretrain', type=str, default='TGA.pth')
parser.add_argument('--image_out', type=str, default='./out/')
opt = parser.parse_args()
gpus_list = range(opt.gpus)
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
print(opt)

def main():
    sys.stdout = Logger(os.path.join(opt.save_test_log,'test_'+systime+'.txt'))
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = False
        torch.cuda.manual_seed(opt.seed)
    pin_memory = True if use_gpu else False 
    # Selecting network
    net = TGA(4) # initial filter generate network
    print("Model size: {:.5f}M".format(sum(p.numel() for p in net.parameters())*4/1048576))
    print('===> {} model has been initialized'.format(opt.model_name))
    filter_net = torch.nn.DataParallel(net, device_ids=gpus_list)
    print('===> load pretrained model')
    if os.path.isfile(opt.pretrain):
        filter_net.load_state_dict(torch.load(opt.pretrain, map_location=lambda storage, loc: storage))
        print('===> pretrained model is load')
    else:
        raise Exception('pretrain model is not exists')
    if use_gpu:
        net = net.cuda(gpus_list[0])

    print('===> Loading test Datasets')
    PSNR_avg = 0
    SSIM_avg = 0
    test_list = ['calendar.txt', 'city.txt', 'foliage.txt', 'walk.txt']
    for test_name in test_list:
        opt.file_test_list = test_name 
        test_set = get_test_set(opt.test_dir, opt.file_test_list, opt.scale, opt.nFrames)
        test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False, pin_memory=pin_memory, drop_last=False)
        print('===> DataLoading Finished')
        PSNR, SSIM = test(test_loader, net, test_name.split('.')[0])
        PSNR_avg += PSNR
        SSIM_avg += SSIM
    PSNR_avg = PSNR_avg/ len(test_list)
    SSIM_avg = SSIM_avg/ len(test_list)
    print('==> Average PSNR = {:.6f}'.format(PSNR_avg))
    print('==> Average SSIM = {:.6f}'.format(SSIM_avg))

    
def test(test_loader, net, test_name):
    train_mode = False
    net.eval()
    count = 0
    avg_psnr = 0
    avg_ssim = 0
    for image_num, data in enumerate(test_loader):
        LR, target, HR = data[0],data[1],data[2]
        with torch.no_grad():
            LR = Variable(LR).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            HR = Variable(HR).cuda(gpus_list[0])
            t0 = time.time()
            prediction = net(LR, HR)
            torch.cuda.synchronize()
            t1 = time.time()
            print("===> Timer: %.4f sec." % (t1 - t0))
        count += 1

        prediction = prediction.squeeze().permute(1,2,0) # [H,W,C]
        prediction = prediction.cpu().clamp(0,1).numpy()[:,:,::-1] # tensor -> numpy, rgb -> bgr 

        target = target.squeeze().permute(1,2,0) # [H,W,C]
        target = target.cpu().data.clamp(0,1).numpy()[:,:,::-1]# tensor -> numpy, rgb -> bgr

        # save image 
        save_img(prediction[8:-8,8:-8,:],test_name,image_num, False)

        # get Y channel 
        prediction_Y = bgr2ycbcr(prediction)
        target_Y = bgr2ycbcr(target)
        #prediction_Y = prediction
        #target_Y = target
        # crop_border
        prediction_Y *= 255
        target_Y *= 255
        prediction_Y, target_Y = crop_border(prediction_Y, target_Y, 8)
        # calculate PSNR and SSIM
        PSNR = calculate_psnr(prediction_Y, target_Y)
        SSIM = calculate_ssim(prediction_Y, target_Y)
        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        print('PSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(PSNR, SSIM))
        avg_psnr += PSNR
        avg_ssim += SSIM
    print('===>{} PSNR = {}'.format(test_name, avg_psnr/(len(test_loader))))
    print('===>{} SSIM = {}'.format(test_name, avg_ssim/(len(test_loader))))
    return (avg_psnr/(len(test_loader)), avg_ssim/(len(test_loader)))

def save_img(prediction,test_name,image_num, att):
    #prediction: bgr [0,1]
    if att == True:
        save_dir = os.path.join(opt.image_out, systime)    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_dir = os.path.join(save_dir, '{}_{:03}'.format(test_name, image_num+1) + '.png')
        cv2.imwrite(image_dir, prediction, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        save_dir = os.path.join(opt.image_out, systime)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_dir = os.path.join(save_dir, '{}_{:03}'.format(test_name, image_num+1) + '.png')
        cv2.imwrite(image_dir, prediction*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    Output:
        type is same as input
        unit8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def crop_border(prediction, target, shave_border=0):
    prediction = prediction[shave_border:-shave_border, shave_border: -shave_border]
    target = target[shave_border: -shave_border, shave_border: -shave_border]
    return prediction, target

def calculate_psnr(prediction, target):
    # prediction and target have range [0, 255]
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



if __name__=='__main__':
    main()
