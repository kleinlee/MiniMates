import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from torch.nn import BatchNorm2d

class DownBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=2)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True,
                 sample_mode='nearest'):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu
        self.sample_mode = sample_mode

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode=self.sample_mode)
        out = self.conv(out)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class ResBlock(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class MotionNet(nn.Module):
    def __init__(self, in_features = [256, 256, 256]):
        super(MotionNet, self).__init__()
        ngf = 12
        n_local_enhancers = 2
        n_downsampling = 2
        n_blocks_local = 1

        # in_features = [256, 256, 256]

        # F1
        f1_model = [
            nn.Conv2d(in_channels=in_features[0], out_channels=ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)]
        f1_model += [
            DownBlock(ngf * 2, ngf * 4, kernel_size=3, padding=1, use_relu=True),
            DownBlock(ngf * 4, ngf * 6, kernel_size=3, padding=1, use_relu=True),
            UpBlock(ngf * 6, ngf * 4, kernel_size=3, padding=1),
            UpBlock(ngf * 4, ngf * 2, kernel_size=3, padding=1)]
        self.f1_model = nn.Sequential(*f1_model)
        self.f1_motion = nn.Conv2d(ngf*2, 2, kernel_size=3, padding=1)

        #f2 and f3
        self.model1_1 = nn.Sequential(*[DownBlock(in_features[1], ngf * 2, kernel_size=3, padding=1)])
        self.model1_2 = nn.Sequential(*[
            DownBlock(ngf * 2, ngf * 4, kernel_size=3, padding=1, use_relu=True),
            DownBlock(ngf * 4, ngf * 8, kernel_size=3, padding=1, use_relu=True),
            ResBlock(ngf * 8, 3, 1),
            UpBlock(ngf * 8, ngf * 6, kernel_size=3, padding=1),
            UpBlock(ngf * 6, ngf * 4, kernel_size=3, padding=1),
            UpBlock(ngf * 4, ngf * 2, kernel_size=3, padding=1)
        ])
        self.model1_3 = nn.Conv2d(ngf * 2, out_channels=2, kernel_size=3, padding=1)

        self.model2_1 = nn.Sequential(*[DownBlock(in_features[2], ngf * 2, kernel_size=3, padding=1)])
        self.model2_2 = nn.Sequential(*[
            DownBlock(ngf * 2, ngf * 4, kernel_size=3, padding=1, use_relu=True),
            DownBlock(ngf * 4, ngf * 8, kernel_size=3, padding=1, use_relu=True),
            ResBlock(ngf * 8, 3, 1),
            UpBlock(ngf * 8, ngf * 4, kernel_size=3, padding=1),
            UpBlock(ngf * 4, ngf * 2, kernel_size=3, padding=1),
            UpBlock(ngf * 2, ngf, kernel_size=3, padding=1)
        ])
        self.model2_3 = nn.Conv2d(ngf, out_channels=2, kernel_size=3, padding=1)

    def forward(self, input1, input2, input3):
        ### output at small scale(f1)
        output_prev = self.f1_model(input1)
        low_motion = self.f1_motion(output_prev)

        ### output at middle scale(f2)
        output_prev = self.model1_2(self.model1_1(input2) + output_prev)
        middle_motion = self.model1_3(output_prev)
        middle_motion = middle_motion + nn.Upsample(scale_factor=2, mode='nearest')(low_motion)

        ### output at large scale(f3)
        output_prev = self.model2_2(self.model2_1(input3) + output_prev)
        high_motion = self.model2_3(output_prev)
        high_motion = high_motion + nn.Upsample(scale_factor=2, mode='nearest')(middle_motion)

        low_motion = low_motion.permute(0, 2, 3, 1)
        middle_motion = middle_motion.permute(0, 2, 3, 1)
        high_motion = high_motion.permute(0, 2, 3, 1)
        return [low_motion, middle_motion, high_motion]


class ResBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features,out_features,1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class UpBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class SameBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features,  kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

def make_coordinate_grid(h, w):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    batch_size = 4
    x = torch.arange(w, dtype=torch.float32)
    y = torch.arange(h, dtype=torch.float32)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.unsqueeze(0)
    meshed = meshed.repeat(batch_size, 1, 1, 1)
    meshed = meshed.permute(0, 3, 1, 2)
    meshed = meshed.to("cuda")
    return meshed

class FreeFace(nn.Module):
    def __init__(self, source_channel,ref_channel, output_size = 384, is_train = True):
        super(FreeFace, self).__init__()
        self.output_size = output_size
        ngf = 16
        if is_train:
            self.init_motion_field_2 = make_coordinate_grid(int(output_size / 4), int(output_size / 4))
            self.init_motion_field_1 = make_coordinate_grid(int(output_size / 8), int(output_size / 8))
            self.init_motion_field_0 = make_coordinate_grid(int(output_size / 16), int(output_size / 16))

            self.init_field_criterion = torch.nn.L1Loss(reduction='mean')
            self.warp_criterion = torch.nn.L1Loss(reduction='mean')
        self.source_encoder = nn.Sequential(
            SameBlock2d(source_channel,ngf*2,kernel_size=3, padding=1),
            DownBlock2d(ngf*2, ngf*2, kernel_size=3, padding=1),
            DownBlock2d(ngf*2,ngf*2,kernel_size=3, padding=1)
        )
        self.ref_encoder = nn.Sequential(
            SameBlock2d(ref_channel,ngf*2,kernel_size=3, padding=1),
            DownBlock2d(ngf*2, ngf*3, kernel_size=3, padding=1),
            SameBlock2d(ngf*3, ngf*3, kernel_size=3, padding=1),
            DownBlock2d(ngf*3, ngf*4,kernel_size=3, padding=1)
        )

        # self.ref_encoder = nn.Sequential(
        #     SameBlock2d(ref_channel,1,kernel_size=3, padding=1),
        #     DownBlock2d(1, 1, kernel_size=3, padding=1),
        #     SameBlock2d(1, 1, kernel_size=3, padding=1),
        #     DownBlock2d(1, ngf*4,kernel_size=3, padding=1)
        # )

        self.decoder = nn.Sequential(
            ResBlock2d(ngf * 6, ngf * 6, 3, 1),
            ResBlock2d(ngf * 6, ngf * 6, 3, 1),
            ResBlock2d(ngf * 6, ngf * 4, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ResBlock2d(64,32, 3, 1),
            # UpBlock2d(32, 32, kernel_size=3, padding=1),
            # nn.Conv2d(32, 4, kernel_size=3, padding=1),
            # nn.Sigmoid()
            nn.Conv2d(32, 4 * (2 * 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Sigmoid()
        )

        self.motion_net = MotionNet(in_features=[ngf*3*2,ngf*3*2,ngf*3*2])


    def calculate_warp_loss(self, input1, input2):
        _, _, h1, w1 = input1.shape
        _, _, h2, w2 = input2.shape
        if h1 != h2 or w1 != w2:
            input2 = F.interpolate(input2, size=(h1, w1), mode='bilinear')
        return self.warp_criterion(input1, input2)

    # @staticmethod
    # def deform_img(inp, deformation):
    #     bs, h, w, _ = deformation.shape
    #     deformation = deformation.permute(0, 3, 1, 2)
    #     deformation = F.interpolate(deformation, size=(384, 384), mode='bilinear')
    #     deformation = deformation.permute(0, 2, 3, 1)
    #     return F.grid_sample(inp, deformation)
    @staticmethod
    def deform_img(inp, deformation):
        bs, h, w, _ = deformation.shape
        if h < 128 or w < 128:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(128, 128), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def ref_input(self, ref_img, ref_prompt):
        ## reference image encoder
        self.ref_img = torch.cat([ref_img, ref_prompt], dim = 1)
        self.ref_in_feature = self.ref_encoder(self.ref_img)
    def interface(self, driving_img, driving_img_face, driving_prompt, bg_mask):
        driving_img = driving_img*bg_mask + driving_img_face*(1-bg_mask)
        self.source_img = torch.cat([driving_img, driving_prompt], dim = 1)
        self.source_in_feature = self.source_encoder(self.source_img)
        motion_net_input = torch.cat([self.source_in_feature, self.ref_in_feature], dim=1)
        # print("motion_net_input", motion_net_input.size())
        # motion_net_input = torch.cat([self.source_img, self.ref_img], dim=1)
        # motion_net_input = F.interpolate(motion_net_input, size=(96, 96), mode='bilinear')
        motion_net_input0 = F.interpolate(motion_net_input, size=(self.output_size//16, self.output_size//16), mode='bilinear')
        motion_net_input1 = F.interpolate(motion_net_input, size=(self.output_size//8, self.output_size//8), mode='bilinear')
        motion_net_input2 = motion_net_input
        self.deformation = self.motion_net(motion_net_input0, motion_net_input1, motion_net_input2)
        ref_trans_feature = F.grid_sample(self.ref_in_feature, self.deformation[2])
        merge_feature = torch.cat([self.source_in_feature, ref_trans_feature],1)
        out = self.decoder(merge_feature)
        return out

    def forward(self, ref_img, ref_prompt, driving_img, driving_img_face, driving_prompt, bg_mask):
        self.ref_input(ref_img, ref_prompt)
        out = self.interface(driving_img, driving_img_face, driving_prompt, bg_mask)
        return out


# import torch
# import time
# import random
# import numpy as np
# import os
# device = "cuda"
# net_g = FreeFace(8, 8).to(device)
# # torch.save(net_g.state_dict(), "DINet_new_medium.pth")
# net_g.eval()
# size = 384
# channel_size = 4
# source_tensor = torch.zeros([1,channel_size,size,size]).to(device)
# source_prompt_tensor = torch.zeros([1,channel_size,size,size]).to(device)
# driving_img_tensor = torch.zeros([1,channel_size,size,size]).to(device)
# driving_img_face_tensor = torch.zeros([1,channel_size,size,size]).to(device)
# driving_prompt_tensor = torch.zeros([1,channel_size,size,size]).to(device)
# bg_mask_tensor = torch.zeros([1,channel_size,size,size]).to(device)
#
# from thop import profile
# from thop import clever_format
# # 9.688G 1.369M   7.804G 1.305M
# flops, params = profile(net_g, inputs=(source_tensor, source_prompt_tensor, driving_img_tensor, driving_img_face_tensor, driving_prompt_tensor, bg_mask_tensor))
# flops, params = clever_format([flops, params], "%.3f")
# print(flops, params)
# start_time = time.time()
#
# with torch.no_grad():
#     net_g.ref_input(source_tensor, source_prompt_tensor)
#     for i in range(2000):
#         print(i, time.time() - start_time)
#         # inference_out, _, _ = net_g(source_tensor, source_lm_tensor, target_lm_tensor, target_bg_tensor)
#         fake = net_g.interface(driving_img_tensor, driving_img_face_tensor, driving_prompt_tensor, bg_mask_tensor)