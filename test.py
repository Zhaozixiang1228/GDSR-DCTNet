# -*- coding: utf-8 -*-

from model import DCTNet
from torch.utils.data import DataLoader
import warnings
from metrics import Rmse
import numpy as np
from scipy.io import savemat
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from utils_gdsr import DRSRH5Dataset, DRSRDataset, save_param, output_img
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

warnings.filterwarnings('ignore') 

def inference_net_eachDataset(dataset_name, net_Path, scale):
    start = time.time()
    # . Get your model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = nn.DataParallel(DCTNet()).to(device)

    if dataset_name == 'Middlebury':
        test_path = r'./DatasetsAfterProcessing/Middlebury_AfterProcessing_'+str(scale)+'X'
    elif dataset_name == 'NYU':
        test_path = r'./DatasetsAfterProcessing/NYU_Test_AfterProcessing_'+str(scale)+'X'
    elif dataset_name == 'Lu':
        test_path = r'./DatasetsAfterProcessing/Lu_AfterProcessing_'+str(scale)+'X'
    elif dataset_name == 'RGBDD':
        test_path = r'./DatasetsAfterProcessing/RGBDD_AfterProcessing_'+str(scale)+'X'

    # # 1. Load the best weight and create the dataloader for testing
    testloader = DataLoader(DRSRDataset(test_path, scale, dataset_name),
                            batch_size=1)
    net.load_state_dict(torch.load(net_Path))
    # 2. Compute the metrics
    metrics = torch.zeros(1, testloader.__len__())
    with torch.no_grad():
        net.eval()
        for i, (Depth, RGB, gt, D_min, D_max) in enumerate(testloader):
            Depth, RGB, gt, D_min, D_max = Depth.cuda(
            ), RGB.cuda(), gt.cuda(), D_min.cuda(), D_max.cuda()
            imgf_raw = net(Depth, RGB).clamp(min=0, max=1)
            imgf = (imgf_raw * (D_max - D_min)) + D_min
            filename = os.path.splitext(
                testloader.dataset.DepthHR_files[i].split('/')[-1])[0]
            if dataset_name == 'Middlebury' or dataset_name == 'Lu':
                imgf2image = output_img(imgf).clip(min=0, max=255)
                gt2image = output_img(gt).clip(min=0, max=255)
            elif dataset_name == 'NYU':
                imgf2image = output_img(imgf)[6:-6, 6:-6]
                gt2image = output_img(gt)[6:-6, 6:-6]
            else:
                imgf2image = output_img(imgf)
                gt2image = output_img(gt)
            metrics[:, i] = Rmse(imgf2image, gt2image)
    end = time.time()
    return metrics.mean(dim=1)


def infrence_all_datasets(net_Path, scale):
    if scale == 'RealScene':
        Rmses = inference_net_eachDataset('RGBDD', net_Path, scale)
    else:
        Rmses = np.zeros(4)
        Rmses[0] = inference_net_eachDataset('Middlebury', net_Path, scale)
        Rmses[1] = inference_net_eachDataset('NYU', net_Path, scale)
        Rmses[2] = inference_net_eachDataset('Lu', net_Path, scale)
        Rmses[3] = inference_net_eachDataset('RGBDD', net_Path, scale)
    return Rmses

def test():
    '''Calculate RMSE value'''
    rmseResults = np.zeros((4,3))
    rmseResults[:, 0] = infrence_all_datasets('models/DCTNet_4X.pth', 4)
    rmseResults[:, 1] = infrence_all_datasets('models/DCTNet_8X.pth', 8)
    rmseResults[:, 2] = infrence_all_datasets('models/DCTNet_16X.pth', 16)
    rmseResults_RealScene1 = infrence_all_datasets(
        'models/DCTNet_4X.pth', 'RealScene')
    rmseResults_RealScene2 = infrence_all_datasets(
        'models/DCTNet_RealScene.pth', 'RealScene')
    
    '''Output the final result'''
    print('==============================================')
    print('The testing RMSE results of Middlebury Dataset')
    print('     X4         X8         X16')
    print('----------------------------------------------')
    print(rmseResults[0,:])
    print('==============================================')
    
    print('==============================================')
    print('The testing RMSE results of NYU V2 Dataset')
    print('     X4         X8         X16')
    print('----------------------------------------------')
    print(rmseResults[1,:])
    print('==============================================')
    
    print('==============================================')
    print('The testing RMSE results of Lu Dataset')
    print('     X4         X8         X16')
    print('----------------------------------------------')
    print(rmseResults[2,:])
    print('==============================================')
    
    print('==============================================')
    print('The testing RMSE results of RGBDD Dataset')
    print('     X4         X8         X16')
    print('----------------------------------------------')
    print(rmseResults[3,:])
    print('==============================================')
    
    print('==============================================')
    print('The testing RMSE results in RealScene RGBDD')
    print('DCTNet in real-world branch')
    print('----------------------------------------------')
    print(rmseResults_RealScene1)
    print('==============================================')
    
    print('==============================================')
    print('The testing RMSE results in RealScene RGBDD')
    print('DCTNet* in real-world branch')
    print('----------------------------------------------')
    print(rmseResults_RealScene2)
    print('==============================================')
    
test()
