import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import torchvision
import math

def rgb2ycbcr(rgb):
    y=rgb[:,0,:,:]*0.299+rgb[:,1,:,:]*0.564+rgb[:,2,:,:]*0.098+16/256
    cb = rgb[:, 0, :, :] *(-0.148) + rgb[:, 1, :, :] * (-0.291) + rgb[:, 2, :, :] * 0.439 + 128/256
    cr = rgb[:, 0, :, :] * 0.439 + rgb[:, 1, :, :] * (-0.368) + rgb[:, 2, :, :] * (-0.071) + 128/256
    return y.unsqueeze(dim=1),cb.unsqueeze(dim=1),cr.unsqueeze(dim=1)

def ycbcr2rgb(y,cb,cr):
    r=1.164*(y-16/256)+1.596*(cr-128/256)
    g=1.164*(y-16/256)-0.813*(cr-128/256)-0.392*(cb-128/256)
    b = 1.164 * (y - 16/256) +2.017 * (cb - 128/256)
    return torch.cat((r,g,b),dim=1)


###the code of calculating SSIM is from https://github.com/jorge-pessoa/pytorch-msssim
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)




def crop_cpu(img,crop_sz,step):# code from ClassSR
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        c,h, w = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[:,x:x + crop_sz, y:y + crop_sz]
            lr_list.append(crop_img)
    return lr_list, num_h, num_w

#color loss
def color_loss(x,y):
    b, c, h, w = x.shape
    true_reflect_view = x.view(b, c, h * w).permute(0, 2, 1)
    pred_reflect_view = y.view(b, c, h * w).permute(0, 2, 1)
    true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
    pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
    cose_value = true_reflect_norm * pred_reflect_norm
    cose_value = torch.sum(cose_value, dim=-1)
    color_loss = torch.mean(1 - cose_value)
    return color_loss

def lf_3(data,pred,gt,model):
    lf_1 = torch.nn.L1Loss()
    lf_2 = SSIM()
    return lf_1(pred*model.module.attention_net(data),gt*model.module.attention_net(data))+1-lf_2(pred*model.module.attention_net(data),gt*model.module.attention_net(data))

def lf_3_single_card(data,pred,gt,model):
    lf_1 = torch.nn.L1Loss()
    lf_2 = SSIM()
    return lf_1(pred*model.attention_net(data),gt*model.attention_net(data))+1-lf_2(pred*model.attention_net(data),gt*model.attention_net(data))

def lf_attention(pred_att,gt,data):
    l1 = torch.nn.MSELoss()
    gt = torch.max(gt, dim=1)
    data = torch.max(data, dim=1)
    gt = torch.clamp(gt[0], 1e-4, 1, out=None)
    z = torch.abs(gt-data[0] ) *(1-gt)
    return l1(z, pred_att)

def lf_noise(pred_noise,gt,data):
    lf2=nn.L1Loss()
    return lf2(torch.max(torch.abs(gt-data)*(1-torch.clamp(data,min=1e-4,max=1)),dim=1)[0], pred_noise)


def cl_loss(p,b,m=3):
    loss=0
    loss+=6*torch.mean(torch.abs(torch.sum(p,dim=1)-b/m))
    return loss*1e-2

class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

def crop_img(img,crop_sz_h,crop_sz_w,step_h,step_w,batch_size):
    [b,c,h, w] = img.shape
    result=torch.zeros([int(b*(math.ceil((h-crop_sz_h)/step_h)+1)*(math.ceil((w-crop_sz_w)/step_w)+1)),c,crop_sz_h,crop_sz_w]).cuda()
    h_space = np.arange(0, h - crop_sz_h + 1, step_h)
    if h_space[-1]!=h-crop_sz_h:
        h_space=np.append(h_space,h - crop_sz_h)
    w_space = np.arange(0, w - crop_sz_w + 1, step_w)
    if w_space[-1]!=w - crop_sz_w:
        w_space=np.append(w_space,w - crop_sz_w)
    index = 0
    num_h = 0
    num_w=0
    count=0
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            result[count*b:(count+1)*b,:,:,:]=img[:,:,x:x + crop_sz_h, y:y + crop_sz_w]
            count=count+1
    # y = torch.zeros([int(result.shape[0] / num_h / num_w), result.shape[1], num_h * num_w, result.shape[2], result.shape[-1]])
    # for i in range(y.shape[0]):
    #     h_list = np.arange(0, result.shape[0], y.shape[0]) + i
    #     y[i, ...] = result[h_list, ...].permute([1, 0, 2, 3])
    return result,h_space,w_space,b

def combine(sr_list,h_space, w_space,b,patch_h,patch_w,step_h,step_w,out_put):
    index=0
    rem=torch.zeros_like(out_put).cuda()
    for i in h_space:
        for j in w_space:
            out_put[:,:,i:i+patch_h,j:j+patch_w]+=sr_list[index*b:(index+1)*b,...]
            rem[:,:,i:i+patch_h,j:j+patch_w]+=1
            index+=1

    out_put=out_put/rem
    return out_put

def get_gt_label(pred,gt,ori_shape):
    lf = nn.MSELoss()
    PSNR = lambda mse: 10 * torch.log10(1. / mse).item() if (mse > 1e-5) else 50
    li_=torch.zeros(gt.shape[0])
    li2_=torch.zeros(gt.shape[0])
    patchs = int(pred.shape[0] / ori_shape)
    idx1 = torch.arange(0, pred.shape[0] - ori_shape, ori_shape).long()
    for i in range(gt.shape[0]):
        # torchvision.utils.save_image(gt[i],'gti.png')
        # torchvision.utils.save_image(pred[i], 'predi.png')
        li2_[i] = PSNR(lf(gt[i], pred[i]))
    for i in range(ori_shape):
        [_,idx]=torch.sort(li2_[idx1],descending=True)
        idx = idx1[idx] + i
        li_[idx[int(patchs*0.6):]]=2
        li_[idx[:int(patchs*0.3)]]=0
        li_[idx[int(patchs*0.3):int(patchs*0.6)]] = 1
    return li_

#according to entropy_of_image
def get_entropy_label(x,ori_shape):
    kernel=1/8*torch.FloatTensor([[1,1,1],[1,0,1],[1,1,1]]).unsqueeze(0).unsqueeze(0).cuda()
    pred=x*255
    tem=pred[:,0,...]*0.299+pred[:,1,...]*0.587+pred[:,2,...]*0.114
    tem2=tem.unsqueeze(dim=1)
    tem=F.conv2d(tem2,kernel,padding=1)
    tem = torch.trunc(tem)
    entropy=[]
    li_=torch.zeros(x.shape[0])
    idx1=torch.arange(0,x.shape[0]-ori_shape,ori_shape).long()
    for i in range(tem.shape[0]):
        bins = torch.histc(tem[i,...].float(), 256, min=0, max=255)
        p=bins/torch.sum(bins)
        p[p==0]=1
        entropy.append(-torch.sum(p*torch.log(p)/math.log(2.0)))
    entropy=torch.Tensor(entropy)
    patchs=int(x.shape[0]/ori_shape)
    for i in range(ori_shape):
        [_, idx] = torch.sort(torch.tensor(entropy[idx1]), descending=True)
        idx=idx1[idx]+i
        li_[idx[int(patchs * 0.75):]] = 0
        li_[idx[:int(patchs * 0.5)]] = 2
        li_[idx[int(patchs * 0.5):int(patchs * 0.75)]] = 1
    return li_


def garma(img,a,b,k):
    return exp(b*(1-k**a))*torch.pow(img,k**a)
def garma_np(img,a,b,k):
    return exp(b*(1-k**a))*np.float_power(img,k**a)

def get_size(ori_shape,num_of_patch):
    h=ori_shape[-2]
    step_h=np.floor(h/num_of_patch)
    patch_h=h-(num_of_patch-1)*step_h

    w = ori_shape[-1]
    step_w = np.floor(w / num_of_patch)
    patch_w = w - (num_of_patch - 1) * step_w

    while True:
        if (step_h<=patch_h*(num_of_patch)/(num_of_patch+1)):
            break
        patch_h+=(num_of_patch-1)
        step_h-=1

    while True:
        if (step_w<=patch_w*(num_of_patch)/(num_of_patch+1)):
            break
        patch_w+=(num_of_patch-1)
        step_w-=1
    return int(patch_h),int(patch_w),int(step_h),int(step_w)
