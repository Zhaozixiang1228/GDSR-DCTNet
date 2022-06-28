# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------

import torchvision.transforms.functional as TF
from torchvision import transforms
from skimage.color import rgb2ycbcr
from skimage.io import imread
from scipy.io import loadmat
import torch.utils.data as Data
import numpy as np
from glob import glob
import torch
import h5py
import os
import json


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def save_param(input_dict, path):
    f = open(path, 'w')
    f.write(json.dumps(input_dict))
    f.close()
    print("Hyper-Parameters have been saved!")


# ----------------------------------------------------------------------------
# Dataset & Image Processing
# ----------------------------------------------------------------------------


def normlization(x):
    # x [N,C,H,W]
    N, C, H, W = x.shape
    m = []
    for i in range(N):
        m.append(torch.max(x[i, :, :, :]))
    m = torch.stack(m, dim=0)[:, None, None, None]
    m = m+1e-10
    x = x/m
    return x, m


def inverse_normlization(x, m):
    return x*m


def im2double(img):
    if img.dtype == 'uint8':
        img = img.astype(np.float32)/255.
    elif img.dtype == 'uint16':
        img = img.astype(np.float32)/65535.
    else:
        img = img.astype(np.float32)
    return img


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def imresize(img, size=None, scale_factor=None):
    # img (np.array) - [C,H,W]
    imgT = torch.from_numpy(img).unsqueeze(0)  # [1,C,H,W]
    if size is None and scale_factor is not None:
        imgT = torch.nn.functional.interpolate(imgT,
                                               scale_factor=scale_factor,
                                               mode='bicubic')
    elif size is not None and scale_factor is None:
        imgT = torch.nn.functional.interpolate(imgT,
                                               size=size,
                                               mode='bicubic')
    else:
        print('Neither size nor scale_factor is given.')
    imgT = imgT.squeeze(0).numpy()
    return imgT


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * \
        0.587000 + img[2:3, :, :] * 0.114000
    return y


def prepare_data_training(data_path,
                          file_name='gdsr_dataset_train',
                          img_size=128,
                          aug=True,
                          scale=4,
                          RGB2Y=False,
                          ):
    # Preparing Training Data from npy docs and save as a h5 file
    # patch_size : the window size of low-resolution images
    # scale : the spatial ratio between low-resolution and guide images
    # train
    print('===========process training data===========')
    I_RGB_Patchs = np.load(os.path.join(data_path, 'I_RGB_Patchs.npy'))
    I_GT_Patchs = np.load(os.path.join(data_path, 'I_GT_Patchs.npy'))
    I_LR_UP_Patchs = np.load(os.path.join(data_path, 'I_LR_UP_Patchs.npy'))

    h5f = h5py.File(os.path.join('.\\data', file_name+'_imgsize_'+str(img_size)+'_scale_'+str(scale)+'_aug_'+str(aug)+'.h5'),
                    'w')

    h5HRdepth = h5f.create_group('HRDepth')
    h5LRdepth = h5f.create_group('LRDepth')
    h5rgb = h5f.create_group('RGB')
    # h5depth_min = h5f.create_group('Depth_min')
    # h5depth_max = h5f.create_group('Depth_max')

    train_num = 0  # 
    for i in range(I_RGB_Patchs.shape[3]):
        I_GT = I_GT_Patchs[:, :, :, i]  # [1,256,256]
        I_Depth = I_LR_UP_Patchs[:, :, :, i]  # [1,256,256]
        I_RGB = I_RGB_Patchs[:, :, :, i]  # [3,256,256]
        # Depth_min = I_GT.min()
        # Depth_max = I_GT.max()

        Depth_files = get_img_file(os.path.join(r'NYU/1-NYU', 'Depth'))
        print("file: %s # samples: %d" % (Depth_files[i], aug*11+(1-aug)*1))

        if RGB2Y:
            pass
        else:  
	        # Original Data
            h5HRdepth.create_dataset(str(train_num),     data=I_GT,
                                     dtype=I_GT.dtype,   shape=I_GT.shape)
            h5rgb.create_dataset(str(train_num),    data=I_RGB,
                                 dtype=I_RGB.dtype,  shape=I_RGB.shape)
            h5LRdepth.create_dataset(str(train_num),  data=I_Depth,
                                     dtype=I_Depth.dtype, shape=I_Depth.shape)


            train_num += 1
    h5f.close()
    if aug:
    	print('training set, # samples %d\n' % (train_num*11))
    else:
        print('training set, # samples %d\n' % (train_num))


class DRSRDataset(Data.Dataset):
    def __init__(self, path, scale, dataset_name, RGB2Y=False):
        self.scale = scale
        self.path = path
        self.DepthHR_files = sorted(get_img_file(path+'/DepthHR'))
        self.RGB_files = sorted(get_img_file(path+'/RGB'))
        self.DepthLR_files = sorted(get_img_file(path+'/DepthLrUp'))
        self.RGB2Y = RGB2Y
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.RGB_files)

    def __getitem__(self, index):
        if self.RGB2Y:
            RGB = rgb2y(np.load(self.RGB_files[index]))
            Depth = np.load(self.DepthLR_files[index])
            if self.dataset_name == 'NYU' or self.dataset_name == 'RGBDD':
                GT = np.load(self.DepthHR_files[index])*100
            elif self.dataset_name == 'Middlebury' or self.dataset_name == 'Lu':
                GT = np.load(self.DepthHR_files[index])
            D_min = GT.min()
            D_max = GT.max()
            return torch.Tensor(Depth), torch.Tensor(RGB), torch.Tensor(GT), torch.tensor(D_min), torch.tensor(D_max)
        else:
            RGB = np.load(self.RGB_files[index])
            Depth = np.load(self.DepthLR_files[index])
            if self.dataset_name == 'NYU' or self.dataset_name == 'RGBDD':
                GT = np.load(self.DepthHR_files[index])*100
            elif self.dataset_name == 'Middlebury' or self.dataset_name == 'Lu':
                GT = np.load(self.DepthHR_files[index])
            D_min = GT.min()
            D_max = GT.max()
            return torch.Tensor(Depth), torch.Tensor(RGB), torch.Tensor(GT), torch.tensor(D_min), torch.tensor(D_max)


class DRSRH5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['HRDepth'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        Depth = np.array(h5f['LRDepth'][key])
        RGB = np.array(h5f['RGB'][key])
        GT = np.array(h5f['HRDepth'][key])
        h5f.close()
        return torch.Tensor(Depth), torch.Tensor(RGB), torch.Tensor(GT)


def output_img(x):
    return x.cpu().detach().numpy()[0, 0, :, :]
