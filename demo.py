#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:12:29 2023
Demo code for "Lateral Strain Imaging using Self-supervised and Physically Inspired Constraints 
in Unsupervised Regularized Elastography"
@author: Ali Kafaei Zad Tehrani
Concordia University, Canada
Part of the code is from the original implementation of the pwc-net irr, https://github.com/visinf/irr

If you use this code please cite the following papers:
[1] Hur J, Roth S. Iterative residual refinement for joint optical flow and occlusion estimation. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2019 (pp. 5754-5763).
[2] Tehrani AK, Rivaz H. MPWC-Net++: Evolution of Optical Flow Pyramidal Convolutional Neural Network for Ultrasound Elastography. SPIE 2021
[3] Tehrani, A.K., Ashikuzzaman, M. and Rivaz, H., 2022. Lateral Strain Imaging using Self-supervised and Physically Inspired Constraints in Unsupervised Regularized Elastography. IEEE Transactions on Medical Imaging.
    
"""


import sys
print(sys.executable)
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.get_device_name(0))
import cv2
from scipy.signal import hilbert
import copy
from scipy import signal

from scipy.ndimage import gaussian_filter 
#%% Some utility functions 
def up_sample_Data(Data,si):
    s=np.shape(Data)
    W=si[1]
    H=si[0]
    res0_test=np.zeros([s[0],s[1],H,W],dtype=np.float32)
    for i in range(s[0]):
        for k in range(s[1]):
            res0_test[i,k,:,:] = cv2.resize(Data[i,k,:,:], dsize=(W, H), interpolation=cv2.INTER_CUBIC)
         
    return res0_test

def hilbert_transform(Im):
    s=np.shape(Im)
    h=np.zeros([s[0],s[1]],dtype=np.float32)
    for i in range(s[1]):
        h[:,i]=np.abs(hilbert(Im[:,i]-np.mean(Im[:,i])))
    return h

def hilbert_transform_imag(Im):
    s=np.shape(Im)
    h=np.zeros([s[0],s[1]],dtype=np.float32)
    for i in range(s[1]):
        mean_line=np.mean(Im[:,i])
        h[:,i]=mean_line+np.imag(hilbert(Im[:,i]-mean_line))
    return h

def data_normalize(Data):
    s=np.shape(Data)
    z=np.zeros([s[0],s[1],s[2],s[3]],dtype=np.float32)  
    for i in range(s[0]):
        for j in range(s[1]):
            z[i,j,:,:]=(Data[i,j,:,:]-np.min(Data[i,j,:,:]))
            z[i,j,:,:]=z[i,j,:,:]/np.max(z[i,j,:,:])
    return z

def Data_form(Im1):
     s=np.shape(Im1)
     Data2=np.zeros([1,3,s[0],s[1]])
     
     N=500.0
     a1=((hilbert_transform_imag(np.array(Im1,copy=True) )))
     a2=np.array(Im1,copy=True)

  
     a2=a2-np.min(a2)
     
     a2=a2/np.max(a2)
     

 

     a1=a1-np.min(a1)
     a1=a1/np.max(a1)
     a3=(hilbert_transform(np.array(a2,copy=True)))
     
     

  

     a3=a3-np.min(a3)
     
     a3=a3/np.max(a3)
     Data2[0,0,:,:]=copy.deepcopy(a1)
     Data2[0,1,:,:]=copy.deepcopy(a2)
     Data2[0,2,:,:]=copy.deepcopy(a3)
     return Data2
 
   
def Strain_Calc(ax_dis,sigma):
    ss=np.shape(ax_dis)

    strain = gaussian_filter(ax_dis, sigma=[sigma[0],sigma[1]],truncate=4)
    strain2=signal.convolve2d(strain, np.array([[1,1,1],[0,0,0],[-1,-1,-1]]),mode='same')
    # strain2=strain2[25:-25,5:-5]
    return strain2


def Strain_Calc_lateral(la_dis,sigma):
    ss=np.shape(la_dis)

    strain = gaussian_filter(la_dis, sigma=[sigma[0],sigma[1]],truncate=4)
    strain2=signal.convolve2d(strain, np.array([[1,0,-1],[1,0,-1],[1,0,-1]]),mode='same')
    # strain2=strain2[25:-25,5:-5]
    return strain2
#%%
model_path = 'Net_Poisson_transform_phantom_upsample_lateral.pth.tar'
import model
net = model.sPICTURE(None)
net.load_state_dict(torch.load(model_path))
net = net.cuda()
A = sio.loadmat('Phantom_Data_Demo.mat')
net.eval()
Adic = {}
Im2 = A['Im2'][:,:,320:-64,32:-32]
Im1 = A['Im1'][:,:,320:-64,32:-32]
s = Im1.shape
up_size = [2048,512]
Adic['input1'] = torch.from_numpy(up_sample_Data(Im2,up_size)).float().cuda()
Adic['input2'] = torch.from_numpy(up_sample_Data(Im1,up_size)).float().cuda()
with torch.no_grad():
    outdic = net(Adic)

flow = outdic['flow_v']
strain_ax = Strain_Calc(flow[0,1,:,:].cpu().data.numpy(),sigma=[26,5])    
strain_la = Strain_Calc_lateral(flow[0,0,:,:].cpu().data.numpy(),sigma=[13,7])
strain_ax = (s[-2]/up_size[0]) * strain_ax[128:-128,64:-64]
strain_la = (s[-1]/up_size[1]) * strain_la[128:-128,64:-64]
strain_la[strain_la>0] = 0
plt.figure();plt.imshow(strain_ax,cmap='hot',aspect='auto');plt.colorbar();plt.title('Axial Strain')
plt.figure();plt.imshow(strain_la,cmap='hot',aspect='auto',vmin=-0.030,vmax=0.0025);plt.colorbar();plt.title('lateral Strain')
