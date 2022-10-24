import shutil
import lpips
import argparse
from torchvision import datasets,transforms
import cv2
import torchvision
from torch.autograd import variable
from torch.utils.data import DataLoader
import torch.nn as nn
from dataloader import Dataset_pair,Dataset_no_pair
from tqdm import tqdm
import sys
import re
import torch
import torch.distributed as dist
import warnings
from utils import *
warnings.filterwarnings('ignore')
from model import *
from torch.utils.data import Dataset
import os
from PIL import Image
from PIL import ImageEnhance
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device_ids = [0]

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_patch',default=5,type=int)
parser.add_argument('--val_root',default='/data1/wuhj/lol_dataset/eval15/low')
parser.add_argument('--val_gt_root',default='/data1/wuhj/lol_dataset/eval15/high')
parser.add_argument('--pretrain_model',default='./pretrain_weights/pretrain_model.pth')
parser.add_argument('--save_pth',default='./result')


with torch.no_grad():
    vgg19=models.vgg19(pretrained=True)
    if torch.cuda.is_available():
        vgg19=vgg19.cuda()
    vgg19=vgg19.features
for idx,param in enumerate(vgg19.parameters()):
    param.requires_grad = False

vgg19_model_new = list(vgg19.children())[:17]
vgg19 = nn.Sequential(*vgg19_model_new)

transforms_imag = torchvision.transforms.Compose([
transforms.RandomCrop([256,256], padding=0),
    torchvision.transforms.ToTensor()])
# path of data
# input_root = '/data1/wuhj/lol_dataset/our485/low'
# label_root = '/data1/wuhj/lol_dataset/our485/high'
# noise_root='/data1/wuhj/lol_dataset/noise_image'
# val_root='/data1/wuhj/lol_dataset/eval15/low'
# val_gt_root='/data1/wuhj/lol_dataset/eval15/high'

# class MyDataset(Dataset):
#     def __init__(self, input_root, label_root, train=True,transform=None,noise_root=None):
#         self.input_root = input_root
#         self.input_files = os.listdir(input_root)
#         if noise_root!=None:
#             self.noise_root = noise_root
#             self.noise_files = os.listdir(noise_root)
#         self.label_root = label_root
#         self.label_files = os.listdir(label_root)
#         self.train=train
#         self.transforms = transform
#
#     def __len__(self):
#         return len(self.input_files)
#
#     def __getitem__(self, index):
#         input_img_path = os.path.join(self.input_root, self.input_files[index])
#         input_img = Image.open(input_img_path)
#
#         label_img_path = os.path.join(self.label_root, self.label_files[index])
#         label_img = Image.open(label_img_path)
#
#         if self.train:
#             noise_img_path = os.path.join(self.noise_root, self.noise_files[index])
#             noise_img = Image.open(noise_img_path)
#             enh_con = ImageEnhance.Contrast(label_img)
#             contrast = 1.35
#             label_img= enh_con.enhance(contrast)
#             seed = np.random.randint(17)
#
#             if self.transforms:
#                 random.seed(seed)
#                 input_img = self.transforms(input_img)
#                 random.seed(seed)
#                 label_img = self.transforms(label_img)
#                 random.seed(seed)
#                 noise_img = self.transforms(noise_img)
#             return (input_img, label_img,noise_img)
#         else:
#             seed = np.random.randint(17)
#             if self.transforms:
#                 random.seed(seed)
#                 input_img = self.transforms(input_img)
#                 random.seed(seed)
#                 label_img = self.transforms(label_img)
#             return (input_img, label_img,self.input_files[index])



def val_psnr(val_loader,model,path,args):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    model = model.eval()
    results=torch.zeros([len(val_loader),3,400,600]).cuda()
    with torch.no_grad():
        all_psnr=0
        with tqdm(total=len(val_loader)) as tq3:
            for idx,(data,gt,name) in enumerate(val_loader):
                te_patch_h, te_patch_w, te_step_h, te_step_w = get_size(data.shape, args.num_of_patch)
                data = data.cuda(device_ids[0])
                gt = gt.cuda(device_ids[0])
                pred,att_map,noise_map,pred_seg = model(data)
                r, num_h, num_w, _ = crop_img(data, te_patch_h, te_patch_w, te_step_h,te_step_w, 1)
                results[idx,...]=pred
                torchvision.utils.save_image(pred,os.path.join(path,'val_{}.png'.format(idx)))
                lf=nn.MSELoss()
                PSNR = lambda mse: 10 * torch.log10(1. / mse).item() if (mse > 1e-5) else 50
                psnr=PSNR(lf(gt,results[idx,...]))
                all_psnr+=psnr
                tq3.update(1)
        print("average {}:{}".format('PSNR',all_psnr/len(val_loader)))

def val_ssim(val_loader,model,path,args):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    model = model.eval()
    results=torch.zeros([len(val_loader),3,400,600]).cuda()
    with torch.no_grad():
        all_ssim=0
        with tqdm(total=len(val_loader)) as tq3:
            for idx,(data,gt,name) in enumerate(val_loader):
                te_patch_h, te_patch_w, te_step_h, te_step_w = get_size(data.shape, args.num_of_patch)
                data = data.cuda(device_ids[0])
                gt = gt.cuda(device_ids[0])
                pred,att_map,noise_map,pred_seg = model(data)
                r, num_h, num_w, _ = crop_img(data, te_patch_h, te_patch_w, te_step_h,te_step_w, 1)
                results[idx,...]=pred
                torchvision.utils.save_image(pred,os.path.join(path,'LOL_val_{}.png'.format(idx)))
                Ssim=SSIM()
                ssim=Ssim(gt,results[idx,...].unsqueeze(dim=0))
                all_ssim+=ssim
                tq3.update(1)
        print("average {}:{}".format('SSIM',all_ssim/len(val_loader)))


def val_lpips(val_loader, model,path,args):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    model = model.eval()
    results = torch.zeros([len(val_loader), 3, 400, 600]).cuda()
    with torch.no_grad():
        all_lpips = 0
        with tqdm(total=len(val_loader)) as tq3:
            for idx, (data, gt, name) in enumerate(val_loader):
                te_patch_h, te_patch_w, te_step_h, te_step_w = get_size(data.shape, args.num_of_patch)
                data = data.cuda(device_ids[0])
                gt = gt.cuda(device_ids[0])
                pred, att_map, noise_map, pred_seg = model(data)
                r, num_h, num_w, _ = crop_img(data, te_patch_h, te_patch_w, te_step_h, te_step_w, 1)
                results[idx, ...] = pred
                torchvision.utils.save_image(pred, os.path.join(path, 'LOL_val_{}.png'.format(idx)))
                lf=lpips.LPIPS().cuda()
                single_lpips=lf.forward(results[idx,...].unsqueeze(dim=0),gt)
                all_lpips += single_lpips.item()
                tq3.update(1)
        print("average {}:{}".format('LPIPS', all_lpips / len(val_loader)))

#for other dataset
def val_other(val_loader, model,path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    model = model.eval()
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as tq3:
            for idx,(data,name) in enumerate(val_loader):
                pad_w = (4 - data.shape[-2] % 4) % 4
                pad_h = (4 - data.shape[-1] % 4) % 4
                data=F.pad(data,[0,int(pad_h),0,int(pad_w)],'replicate')
                data = data.cuda()
                pred,_,_,_ = model(data)
                pred=pred[:,:,0:pred.shape[-2]-pad_w,0:pred.shape[-1]-pad_h]
                # results[idx,...]=pred
                torchvision.utils.save_image(pred,path+"/"+name[0])
                tq3.update(1)


if __name__=='__main__':
    args = parser.parse_args()
    dataset_val=Dataset_pair(args.val_root,args.val_gt_root,train=False,transform=torchvision.transforms.ToTensor())
    val_loader=torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, drop_last=False)
    model=IEDCN()
    if torch.cuda.is_available():
        # model = model.cuda(device_ids[0])
        model = model.cuda()
    model.load_state_dict(torch.load(args.pretrain_model))
    # model = nn.DataParallel(model, device_ids)

    val_psnr(val_loader, model,args.save_pth,args)
    # val_ssim(val_loader,model,args)
    # val_lpips(val_loader, model,args)
    # val_other(val_loader, model,save_path,args)