#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:04:10 2023
Network for "Lateral Strain Imaging using Self-supervised and Physically Inspired Constraints 
in Unsupervised Regularized Elastography"
@author: Ali Kafaei Zad Tehrani
Concordia University, Canada
Part of the code is from the original implementation of the pwc-net irr, https://github.com/visinf/irr

If you use this code please cite the following papers:
[1] Hur J, Roth S. Iterative residual refinement for joint optical flow and occlusion estimation. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2019 (pp. 5754-5763).
[2] Tehrani AK, Rivaz H. MPWC-Net++: Evolution of Optical Flow Pyramidal Convolutional Neural Network for Ultrasound Elastography. SPIE 2021
[3] Tehrani, A.K., Ashikuzzaman, M. and Rivaz, H., 2022. Lateral Strain Imaging using Self-supervised and Physically Inspired Constraints in Unsupervised Regularized Elastography. IEEE Transactions on Medical Imaging.
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging
import numpy as np
#%% Defining some functions
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )
def conv2(in_planes, out_planes, kernel_size=(5,3), stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=(2,1), bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)


def rescale_flow(flow, div_flow, width_im, height_im, to_local=True):
    if to_local:
        u_scale = float(flow.size(3) / width_im / div_flow)
        v_scale = float(flow.size(2) / height_im / div_flow)
    else:
        u_scale = float(width_im * div_flow / flow.size(3))
        v_scale = float(height_im * div_flow / flow.size(2))

    u, v = flow.chunk(2, dim=1)
    u = u * u_scale
    v = v * v_scale

    return torch.cat([u, v], dim=1)

def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False).cuda()
    return grids_cuda


class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
        flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)        
        x_warp = tf.grid_sample(x, grid, align_corners=True)
        device1 = x.device

        mask = torch.ones(x.size(), requires_grad=False).cuda(device = device1)
        mask = tf.grid_sample(mask, grid, align_corners=True)
        mask = (mask >= 1.0).float()

        return x_warp * mask
    
def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


def compute_cost_volume(feat1, feat2, param_dict):
    """
    only implemented for:
        kernel_size = 1
        stride1 = 1
        stride2 = 1
    """

    max_disp = param_dict["max_disp"]

    _, _, height, width = feat1.size()
    num_shifts = 2 * max_disp + 1
    feat2_padded = tf.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)

    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            corr = torch.mean(feat1 * feat2_padded[:, :, i:(height + i), j:(width + j)], axis=1, keepdims=True)
            cost_list.append(corr)
    cost_volume = torch.cat(cost_list, axis=1)
    return cost_volume

#%% 
""""
Definition of pwc modules
Some functions are from original implementation of pwc-net irr, https://github.com/visinf/irr
"""
class FeatureExtractorpp(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractorpp, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()
        #layer = nn.Sequential(
        #conv(num_chs[0], num_chs[1], stride=1),
        #conv(num_chs[1], num_chs[1]))
        #self.convs.append(layer)
        for l, (ch_in, ch_out) in enumerate(zip(num_chs[0:-1], num_chs[1:])):
            if (l==0):
                layer = nn.Sequential(
                conv2(ch_in, ch_out,stride=1),
                torch.nn.AvgPool2d(2,stride=2,ceil_mode=True),
                conv(ch_out, ch_out)
                )
            elif (l==1):
                layer = nn.Sequential(
                conv(ch_in, ch_out,stride=1),
                #torch.nn.AvgPool2d(2,stride=2,ceil_mode=True),
                conv(ch_out, ch_out)
                )
            else:
                layer = nn.Sequential(
                conv(ch_in, ch_out,stride=1),
                torch.nn.AvgPool2d(2,stride=2,ceil_mode=True),
                conv(ch_out, ch_out)
                )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]

class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out

class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)
#%%
"""
Definition of mpwcnet++ (MPWCNet2 [2]) and sPICTURE [3]
"""
class MPWCNet2(nn.Module):
    def __init__(self, args, div_flow=0.1):
        super(MPWCNet2, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 5
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractorpp(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])
        # self.refine = RefineFlow2(35)

        initialize_msra(self.modules())
        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}


    def forward(self, input_dict):

        x1_raw = input_dict['input1']
        x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        output_dict = {}
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = upsample2d_as(flow, x1, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)

            # correlation
            #out_corr = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr = compute_cost_volume(x1, x2_warp, self.corr_params)
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=True)

            x1_1by1 = self.conv_1x1[l](x1)
            x_intm, flow_res = self.flow_estimators(torch.cat([out_corr_relu, x1_1by1, flow], dim=1))
            flow = flow + flow_res

            flow_fine = self.context_networks(torch.cat([x_intm, flow], dim=1))
            flow = flow + flow_fine

            flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=False)
            # if (l==4):
            #     x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)
            #     my_refine = self.refine(flow,x1-x2_warp,x1) 
            #     flows.append(my_refine)
            # else:
            flows.append(flow)

            # upsampling or post-processing
            if l == self.output_level:
                break
        
        flow1 = upsample2d_as(flow, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
        output_dict['flow'] = flows
        output_dict['flow1'] = flow1
        output_dict['x2'] = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)
        output_dict['x1'] = x1
        output_dict['x2_raw'] = self.warping_layer(x2_raw, flow1, height_im, width_im, 1)
        return output_dict





class sPICTURE(nn.Module):
    def __init__(self, args, div_flow=0.1,known_operator=0):
        super(sPICTURE, self).__init__()
        self.MPWCNet = MPWCNet2(None)



    def forward(self, input_dict):

        self.warping_layer = WarpingLayer()
        x1_raw = input_dict['input1']
        x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        
        x_int = self.MPWCNet(input_dict)
        flow_pre_all = x_int['flow']
        for k in range(len(flow_pre_all)):
            flow_pre_all[k] = 10*upsample2d_as(flow_pre_all[k], x1_raw, mode="bilinear")
        x1 = upsample2d_as(x_int['x1'], x1_raw,mode="bilinear")
        x2 = upsample2d_as(x_int['x2'], x2_raw,mode="bilinear")
        flow_pre = flow_pre_all[-1]
        axial = flow_pre[:,1,:,:].unsqueeze(1)
        lateral = flow_pre[:,0,:,:].unsqueeze(1)
        
        
        
        axialn = 1*(axial)
        lateraln = 1*(lateral)
 
        flow_v = torch.cat([lateraln, axialn], dim=1)

        output_dict = {}




        
        output_dict['flow_v'] = flow_v
        
                
        output_dict['x2'] = self.warping_layer(x2, flow_pre, height_im, width_im, 1)
        output_dict['x1'] = x1
        output_dict['x2_raw'] = self.warping_layer(x2_raw, flow_pre, height_im, width_im, 1)
        output_dict['x1_raw'] = x1_raw
        output_dict['x2_raw_v'] = self.warping_layer(x2_raw, flow_v, height_im, width_im, 1)

   

        return output_dict