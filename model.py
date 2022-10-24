import torchvision
import torch.nn as nn
import time
import os
import shutil
from torchvision import models
from utils import *
import torch
out_channel=16

class Deepse_conv(nn.Module):
    def __init__(self,in_c,out_c):
        super(Deepse_conv, self).__init__()
        self.d_conv=nn.Conv2d(in_c,in_c,3,padding=1,groups=in_c)
        self.s_conv=nn.Conv2d(in_c,out_c,1,padding=0)
        self.bn=nn.BatchNorm2d(in_c)
        self.relu=nn.ReLU6()
    def forward(self,x):
        return self.s_conv(self.relu(self.bn(self.d_conv(x))))

class Deepse_conv2(nn.Module):
    def __init__(self, in_c):
        super(Deepse_conv2, self).__init__()
        # self.s1_conv = nn.Conv2d(in_c, in_c*6, 1, padding=0)
        self.d_conv = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c)
        self.s_conv = nn.Conv2d(in_c, in_c, 1, padding=0)
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU6()

    def forward(self, x):
        return x + self.s_conv(self.relu(self.bn(self.d_conv(x))))

class NoiseNet(nn.Module):
    def __init__(self):
        super(NoiseNet, self).__init__()
        times=2
        self.conv = nn.Sequential(nn.Conv2d(4, 8*times, (3, 3), padding=1), nn.ReLU(),
                                   nn.Conv2d(8*times, 16*times, (3, 3), padding=1), nn.ReLU(),
                                   # nn.Conv2d(16*times, 16*times, (3, 3), padding=1), nn.ReLU(),
                                   # nn.Conv2d(16*times, 16*times, (3, 3), padding=1), nn.ReLU(),
                                   nn.Conv2d(16*times, 8*times, (3, 3), padding=1), nn.ReLU(),
                                   nn.Conv2d(8*times, 1, (3, 3), padding=1), nn.ReLU())
    def forward(self, x):
        return self.conv(x)

class Conv(nn.Module):
    def __init__(self, in_c,out_c):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, in_c-1, 3,dilation=2, padding=2),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(in_c - 1),
                                  nn.Conv2d(in_c-1, in_c-1, 3, dilation=2, padding=2),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(in_c-1),
                                  nn.Conv2d(in_c-1, in_c-1, 3, dilation=2, padding=2),
                                  nn.BatchNorm2d(in_c - 1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(in_c - 1),
                                  nn.Conv2d(in_c-1, out_c, 3, dilation=2, padding=2),
                                  nn.ReLU())
    def forward(self, x):
        return self.conv(x)

class Resblock_ds(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(Resblock_ds, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(in_channel, mid_channel, (3, 3), padding=1),
        #                            nn.ReLU())
        self.conv2 = nn.Sequential(Deepse_conv2(mid_channel),
                                   nn.ReLU(),
                                   Deepse_conv2(mid_channel),
                                   nn.ReLU())

    def forward(self, x):
        # x1 = self.conv1(x)
        return x+ self.conv2(x)


class Resnet(nn.Module):
    def __init__(self, in_c,out_c):
        super(Resnet, self).__init__()
        self.name = 'resnet'
        self.conv = nn.Sequential(
                                  # nn.BatchNorm2d(in_c),
                                  nn.Conv2d(in_c,in_c-1,3,padding=1),
                                  Resblock_ds(in_c-1, in_c-1),
                                  # Resblock(in_c-1, in_c-1),
                                  Resblock_ds(in_c-1, (in_c-1)*1),
                                  Resblock_ds((in_c-1)*1,(in_c-1)*1),
                                  Resblock_ds((in_c-1)*1, in_c-1),
                                  nn.Conv2d(in_c-1,out_c,3,padding=1))
    def forward(self, x):
        return self.conv(x)


class U_net(nn.Module):
    def __init__(self, in_c,out_c):
        super(U_net, self).__init__()
        self.name = 'unet'
        self.conv_l1 = nn.Sequential(nn.Conv2d(in_c, 8*8, (3, 3), padding=1),nn.BatchNorm2d(8*8), nn.ReLU(), nn.Conv2d(8*8, 8*8, 1))
        self.conv_l2 = nn.Sequential(nn.Conv2d(8*8, 16*8, (3, 3), padding=1), nn.BatchNorm2d(16*8),nn.ReLU(), nn.Conv2d(16*8, 16*8, 1))
        self.conv_l3 = nn.Sequential(nn.Conv2d(16*8, 32*8, (3, 3), padding=1), nn.BatchNorm2d(32*8),nn.ReLU(), nn.Conv2d(32*8, 32*8, 1))
        self.conv_r3 = nn.Sequential(nn.Conv2d(32*8, 16*8, (3, 3), padding=1), nn.BatchNorm2d(16*8),nn.ReLU(), nn.Conv2d(16*8, 16*8, 1))
        self.conv_r2 = nn.Sequential(nn.Conv2d(32*8, 16*8, (3, 3), padding=1), nn.BatchNorm2d(16*8),nn.ReLU(), nn.Conv2d(16*8, 8*8, 1))
        self.conv_r1 = nn.Sequential(nn.Conv2d(16*8, 8*8, (3, 3), padding=1), nn.BatchNorm2d(8*8),nn.ReLU(), nn.Conv2d(8*8, out_c, 1))
        self.down_1 = nn.MaxPool2d((2, 2), 2, return_indices=True)
        self.down_2 = nn.MaxPool2d((2, 2), 2, return_indices=True)
        # self.down_for_y=nn.MaxPool2d((2, 2), 2, return_indices=False)
        self.up_2 = nn.MaxUnpool2d((2, 2), 2)
        self.up_1 = nn.MaxUnpool2d((2, 2), 2)
        # self.conv1d=nn.Conv1d(1,1,5,padding=2)
    def forward(self, x):
            x = self.conv_l1(x)
            y2, ind1 = self.down_1(x)
            y2 = self.conv_l2(y2)
            y3, ind2 = self.down_2(y2)
            y3=self.conv_l3(y3)
            y3 = self.conv_r3(y3)
            y3 = self.up_2(y3, ind2)
            y3 = torch.cat((y2, y3), dim=1)
            y3 = self.conv_r2(y3)
            y3 = self.up_1(y3, ind1)
            y3 = self.conv_r1(torch.cat((y3, x), dim=1))
            return y3


class U_net_light3(nn.Module):
    def __init__(self, in_c,out_c,times=4):
        super(U_net_light3, self).__init__()
        self.name = 'unet'
        self.conv_l1 = nn.Sequential(nn.BatchNorm2d(int(in_c)), Deepse_conv2(int(in_c)),nn.ReLU(),
                                     nn.BatchNorm2d(int(in_c)), nn.Conv2d(int(in_c), int(8 * times), 1),nn.ReLU())
        self.conv_l2 = nn.Sequential(nn.BatchNorm2d(int(8 * times)),
                                     Deepse_conv2(int(8 * times)), nn.ReLU(),
                                     nn.BatchNorm2d(int(8 * times)), nn.Conv2d(int(8 * times), int(16 * times), 1),
                                     nn.ReLU())
        self.conv_l3 = nn.Sequential(nn.BatchNorm2d(int(16 * times)),
                                     Deepse_conv2(int(16 * times)), nn.ReLU(),
                                     nn.BatchNorm2d(int(16 * times)), nn.Conv2d(int(16 * times), int(32 * times), 1),
                                     nn.ReLU())
        self.conv_r3 = nn.Sequential(nn.BatchNorm2d(int(32 * times)),
                                     Deepse_conv2(int(32 * times)), nn.ReLU(),
                                     nn.BatchNorm2d(int(32 * times)), nn.Conv2d(int(32 * times), int(16 * times), 1),
                                     nn.ReLU())
        self.conv_r2 = nn.Sequential(nn.BatchNorm2d(int(32 * times)),
                                     Deepse_conv2(int(32 * times)), nn.ReLU(),
                                     nn.BatchNorm2d(int(32 * times)), nn.Conv2d(int(32 * times), int(8 * times), 1),
                                     nn.ReLU())
        self.conv_r1 = nn.Sequential(nn.BatchNorm2d(int(16 * times)),
                                     Deepse_conv2(int(16 * times)), nn.ReLU(),
                                     nn.BatchNorm2d(int(16 * times)), nn.Conv2d(int(16 * times), int(out_c), 1),
                                     nn.ReLU())
        self.down_1 = nn.MaxPool2d((2, 2), 2, return_indices=True)
        self.down_2 = nn.MaxPool2d((2, 2), 2, return_indices=True)
        self.up_2 = nn.MaxUnpool2d((2, 2), 2)
        self.up_1 = nn.MaxUnpool2d((2, 2), 2)
    def forward(self, x):
            x = self.conv_l1(x)
            y2, ind1 = self.down_1(x)
            y2 = self.conv_l2(y2)
            y3, ind2 = self.down_2(y2)
            y3=self.conv_l3(y3)
            y3 = self.conv_r3(y3)
            y3 = self.up_2(y3, ind2)
            y3 = torch.cat((y2, y3), dim=1)
            y3 = self.conv_r2(y3)
            y3 = self.up_1(y3, ind1)
            y3 = self.conv_r1(torch.cat((y3, x), dim=1))
            return y3


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(256, 3)
        self.relu=nn.ReLU()
        self.CondNet =nn.Sequential(nn.Conv2d(4,32,4,padding=4), nn.LeakyReLU(0.1, True),
                                    nn.Conv2d(32, 64, 1), nn.LeakyReLU(0.1, True),
                                    nn.MaxPool2d((2, 2), 2, return_indices=False),
                                    nn.Conv2d(64, 128, 1), nn.LeakyReLU(0.1, True),
                                    nn.MaxPool2d((2, 2), 2, return_indices=False),
                                    nn.Conv2d(128,256, 1), nn.LeakyReLU(0.1, True),
                                    # nn.MaxPool2d((2, 2), 2, return_indices=False),
                                    )
    def forward(self, x):
        x = self.CondNet(x)
        x = nn.AvgPool2d(x.size()[2])(x)
        x = x.view(x.size(0), -1)
        x = self.lastOut(x)
        x=self.relu(x)
        # x=torch.clamp(x,1e-7)
        return x/torch.sum(x, dim=1).unsqueeze(dim=-1).expand_as(x)


class Concate(nn.Module):
    def __init__(self,branch_num):
        super(Concate, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(int(out_channel *branch_num), int(out_channel *branch_num),1),nn.BatchNorm2d(int(out_channel *branch_num)))
        self.conv2=nn.Sequential(nn.Conv2d(int(out_channel *branch_num), int(out_channel *branch_num),3,padding=1),nn.BatchNorm2d(int(out_channel *branch_num)))
        self.conv3 = nn.Conv1d(1,1,7,padding=3)
        self.conv4=nn.Sequential(nn.Conv2d(int(out_channel * branch_num),3,3,padding=1),nn.ReLU())
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.sig=nn.Sigmoid()
        self.relu=nn.ReLU()
    def forward(self, x):
        z=self.pool(x).squeeze()
        z=z.unsqueeze(dim=-2)
        if len(z.shape)==2:
            z=z.unsqueeze(dim=0)
        z=self.conv3(z).transpose(-1, -2).unsqueeze(dim=-1)
        # if len(z.shape) == 1:
        #     z.unsqueeze(0)
        # z=self.conv3(z).unsqueeze(-1).unsqueeze(-1)
        return self.conv4(self.relu((self.conv1(x)+self.conv2(x))*self.sig(z)))#

class IEDCN(nn.Module):
    def __init__(self):
        super(IEDCN, self).__init__()
        self.branch2 = U_net_light3(66, int(out_channel), 8)
        self.branch4 = Resnet(66, int(out_channel))
        self.branch5 = Conv(66, int(out_channel))
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(64,64, 3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        self.concat1 = Concate(1)
        self.concat2 = Concate(2)
        self.concat3 = Concate(3)
        self.attention_net = U_net_light3(3, 1, 1)
        self.noise_net = NoiseNet()

        self.enhancer = nn.Sequential(nn.BatchNorm2d(6),nn.Conv2d(6, 64, 3, padding=1), nn.ReLU(),
                                      nn.BatchNorm2d(64),nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                      nn.BatchNorm2d(64),nn.Conv2d(64, 3, 3, padding=1), nn.ReLU())
    def forward(self,x):
        att_map = self.attention_net(x)
        noise_map = self.noise_net(torch.cat((x, att_map), dim=1))
        tr_patch_h, tr_patch_w, tr_step_h, tr_step_w = get_size(x.shape, 5)
        r, num_h, num_w, _ = crop_img(torch.cat((x,att_map,noise_map),dim=1), tr_patch_h, tr_patch_w, tr_step_h, tr_step_w, 1)
        data_af=r[:,0:3,...]
        att_after=r[:,3,...].unsqueeze(dim=1)
        noise_after=r[:,4,...].unsqueeze(dim=1)
        result_seg=torch.zeros_like(data_af).cuda()
        out_put = torch.zeros_like(x).cuda()
        cl_result = get_entropy_label(data_af, x.shape[0])
        list_1 = (cl_result == 0).nonzero().squeeze(dim=-1)
        list_2 = (cl_result == 1).nonzero().squeeze(dim=-1)
        list_3 = (cl_result == 2).nonzero().squeeze(dim=-1)
        list_all = [list_1, list_2, list_3]
        x1 = self.conv1(data_af)
        x1_5 = self.conv2(x1)
        b2=self.branch2(torch.cat((x1_5,att_after,noise_after),dim=1))
        if len(list_all[1])+len(list_all[2]) !=0:
            tem=self.conv4(x1_5[torch.cat((list_all[1],list_all[2])),...])
            b4=self.branch4(torch.cat((tem,att_after[torch.cat((list_all[1],list_all[2])),...],noise_after[torch.cat((list_all[1],list_all[2])),...]),dim=1))
        if len(list_all[0]) != 0:
            result_seg[list_all[0],...] = self.concat1(b2[list_all[0],...])
        if len(list_all[1]) != 0:
            result_seg[list_all[1],...] =self.concat2(torch.cat((b2[list_all[1],...],b4[:len(list_all[1]),...]),dim=1))
        if len(list_all[2]) != 0:
            result_seg[list_all[2],...] =self.concat3(torch.cat((b2[list_all[2],...],b4[len(list_all[1]):,...],self.branch5(torch.cat((self.conv5(tem[len(list_all[1]):,...]),att_after[list_all[2],...],noise_after[list_all[2],...]),dim=1))),dim=1))
        predtem = combine(result_seg, num_h, num_w, x.shape[0], tr_patch_h, tr_patch_w, tr_step_h, tr_step_w, out_put)
        pred=self.enhancer(torch.cat((predtem,x),dim=1))
        return pred,att_map,noise_map,[predtem,result_seg]
