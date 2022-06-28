# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import kornia
from dct import dct_2d,idct_2d

import torch.nn.functional as F
import sys
sys.path.append("..")


class Weight_Prediction_Network(nn.Module):
    def __init__(self,n_feats=64):
        super(Weight_Prediction_Network, self).__init__()
        f = n_feats // 4 
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv_dilation = nn.Conv2d(f, f, kernel_size=3, padding=1, 
                                        stride=3, dilation=2)    
    def forward(self, x): # x is the input feature  
        x = self.conv1(x)
        shortCut = x
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=7, stride=3)
        x = self.relu(self.conv_max(x))
        x = self.relu(self.conv3(x))
        x = self.conv3_(x)
        x = F.interpolate(x, (shortCut.size(2), shortCut.size(3)),
                          mode='bilinear', align_corners=False)
        shortCut = self.conv_f(shortCut)
        x = self.conv4(x+shortCut)
        x = self.sigmoid(x)
        return x
     
class Coupled_Layer(nn.Module):
    def __init__(self,
                 coupled_number=32,
                 n_feats=64,
                 kernel_size=3):
        super(Coupled_Layer, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size
        self.kernel_shared_1=nn.Parameter(nn.init.kaiming_uniform(torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_1=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_1=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_shared_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        
        self.bias_shared_1=nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_1=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        self.bias_rgb_1=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        
        self.bias_shared_2=nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_2=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        self.bias_rgb_2=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        
    def forward(self, feat_dlr, feat_rgb):
        shortCut = feat_dlr
        feat_dlr = F.conv2d(feat_dlr,
                                  torch.cat([self.kernel_shared_1, self.kernel_depth_1], dim=0), 
                                  torch.cat([self.bias_shared_1, self.bias_depth_1], dim=0),
                                  padding=1)
        feat_dlr = F.relu(feat_dlr, inplace=True)
        feat_dlr = F.conv2d(feat_dlr,
                                  torch.cat([self.kernel_shared_2, self.kernel_depth_2], dim=0), 
                                  torch.cat([self.bias_shared_2, self.bias_depth_2], dim=0),
                                  padding=1)
        feat_dlr = F.relu(feat_dlr + shortCut, inplace=True)
        shortCut = feat_rgb
        feat_rgb = F.conv2d(feat_rgb,
                                  torch.cat([self.kernel_shared_1, self.kernel_rgb_1], dim=0), 
                                  torch.cat([self.bias_shared_1, self.bias_rgb_1], dim=0),
                                  padding=1)
        feat_rgb = F.relu(feat_rgb, inplace=True)
        feat_rgb = F.conv2d(feat_rgb,
                                  torch.cat([self.kernel_shared_2, self.kernel_rgb_2], dim=0), 
                                  torch.cat([self.bias_shared_2, self.bias_rgb_2], dim=0),
                                  padding=1)
        feat_rgb = F.relu(feat_rgb + shortCut, inplace=True)
        return feat_dlr, feat_rgb
    
class Coupled_Encoder(nn.Module):
    def __init__(self,
                 n_feat=64,
                 n_layer=4):
        super(Coupled_Encoder, self).__init__()
        self.n_layer = n_layer
        self.init_deep=nn.Sequential( 
                nn.Conv2d(1, n_feat, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
                nn.ReLU(True),                               
                )  
        self.init_rgb=nn.Sequential( 
                nn.Conv2d(3, n_feat, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
                nn.ReLU(True),                               
                )             
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer() for i in range(self.n_layer)])   

    def forward(self, feat_dlr, feat_rgb):
        feat_dlr = self.init_deep(feat_dlr)
        feat_rgb = self.init_rgb(feat_rgb)
        for layer in self.coupled_feat_extractor:
            feat_dlr, feat_rgb = layer(feat_dlr, feat_rgb)
        return feat_dlr, feat_rgb

 
class Decoder_Deep(nn.Module):
    def __init__(self,
                 n_feats=64):  
        super(Decoder_Deep, self).__init__()
        self.Decoder_Deep=nn.Sequential( 
                nn.Conv2d(n_feats, n_feats//2, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
                nn.ReLU(True),                    
                nn.Conv2d(n_feats//2, n_feats//4, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
                nn.ReLU(True), 
                nn.Conv2d(n_feats//4, 1, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
                nn.ReLU(True),                
                )   
    def forward(self, x):
        return self.Decoder_Deep(x)  
    
class DCTNet(nn.Module):
    ''' 
    Solver for the problem: min_{x} |x-d|_2^2+lambd|L(x)-L(r).*w|_2^2
    d - input low-resolution image
    r - guidance image (we want transfer the gradient of r into d)
        input RGB image
    z - output super-resolution image 
    L - Laplacian operator
    w - Edge weight matrix (to be learned by WeightLearning Network)
        *Note: the solution of this problem is idct(p/c)
               p = dct(lambd*L(L(r)).*w + d)
               c = lambd*K^2+1
               K = self.get_K()
    '''
    def __init__(self, lambd=3., n_feats=64):
        super(DCTNet, self).__init__()
        self.n_feats = n_feats
        self.lambd = nn.Parameter(
                torch.nn.init.normal(
                        torch.full(size=(1,self.n_feats,1,1),fill_value=lambd),mean=0.1,std=0.3))
        #                torch.nn.init.kaiming_normal(
#                        torch.full(size=(1,self.n_feats,1,1),fill_value=lambd)))
        self.WPNet = Weight_Prediction_Network()

        self.Encoder_coupled = Coupled_Encoder()
        self.Decoder_depth = Decoder_Deep()
       
    def get_K(self, H, W, dtype, device):
        pi = torch.acos(torch.Tensor([-1]))
        cos_row = torch.cos(pi*torch.linspace(0,H-1,H)/H).unsqueeze(1).expand(-1,W)
        cos_col = torch.cos(pi*torch.linspace(0,W-1,W)/W).unsqueeze(0).expand(H,-1)
        kappa = 2*(cos_row+cos_col-2)
        kappa = kappa.to(dtype).to(device)
        return kappa[None,None,:,:] # shape [1,1,H,W]
    
    def get_Lap(self, dtype, device):
        laplacian = kornia.filters.Laplacian(3) 
        f=laplacian
        return f
    
    def forward(self, x, y):
        # x - input depth image d, shape [N,C,H,W]
        # y - guidance RGB image r, shape [N,1,H,W] or [N,C,H,W]

        if len(y.shape)==3:
            y = y[:,None,:,:] 
        N,C,H,W = x.shape   
            
        high_Dim_D, high_Dim_R = self.Encoder_coupled(x, y)
        
        # get weight
        weight=self.WPNet(high_Dim_R)
        # weight=self.WPNet(y)
        
        # get SR image (64 channel)
        lambd = torch.exp(self.lambd).to(x.device)
        k2 = self.get_K(H, W, x.dtype, x.device).pow(2) 
        L = self.get_Lap(x.dtype, x.device)
        # P = dct_2d(
        #         torch.mul(lambd*L(L(high_Dim_R)),weight)+high_Dim_D
        #         )
        P = dct_2d(
            lambd*L(torch.mul(L(high_Dim_R), weight))+high_Dim_D
                )
        C = lambd*k2+1
        z = idct_2d(P/C)

        SR_deepth = self.Decoder_depth(z)
        return SR_deepth
