import shutil
import random
from torchvision import datasets,transforms
import torchvision
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from tqdm import tqdm
import warnings
from utils import *
warnings.filterwarnings('ignore')
from model import *
from dataloader import Dataset_pair
from torch.utils.data import Dataset
import os

from utils import *
import argparse
#settings
parser = argparse.ArgumentParser()
parser.add_argument('--num_of_patch',default=5,type=int)
parser.add_argument('--lr',default=2e-4)
parser.add_argument('--epochs',default=400)
parser.add_argument('--batchsize',default=4)
parser.add_argument('--model_save_psnr_st',default=19.5)
parser.add_argument('--input_root',default='/data1/wuhj/lol_dataset/our485/low')
parser.add_argument('--label_root',default='/data1/wuhj/lol_dataset/our485/high')
parser.add_argument('--noise_root',default='./noise_image')
parser.add_argument('--val_root',default='/data1/wuhj/lol_dataset/eval15/low')
parser.add_argument('--val_gt_root',default='/data1/wuhj/lol_dataset/eval15/high')
parser.add_argument('--save_pth',default='./result')

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device_ids = [0]

# load vgg network for percetual loss
with torch.no_grad():
    vgg19=models.vgg19(pretrained=True)
    if torch.cuda.is_available():
        vgg19=vgg19.cuda()
    vgg19=vgg19.features
for idx,param in enumerate(vgg19.parameters()):
    param.requires_grad = False

vgg19_model_new = list(vgg19.children())[:17]
vgg19 = nn.Sequential(*vgg19_model_new)


def val(val_loader,model,ep,pth,max_psnr,args):
    model = model.eval()
    results=torch.zeros([len(val_loader),3,400,600])
    with torch.no_grad():
        all_psnr=0
        with tqdm(total=len(val_loader)) as tq3:
            for idx,(data,gt,name) in enumerate(val_loader):
                data = data.cuda(device_ids[0])
                gt = gt.cuda(device_ids[0])
                tq3.update(1)
                pred,_,_,_=model(data)
                results[idx,...]=pred
                lf=nn.MSELoss()
                PSNR = lambda mse: 10 * torch.log10(1. / mse).item() if (mse > 1e-5) else 50
                psnr=PSNR(lf(gt,pred))
                all_psnr+=psnr
        if (all_psnr/len(val_loader) > max_psnr and all_psnr/len(val_loader)>args.model_save_psnr_st) or ep==args.epochs-1:
            for idx, (data, gt,name) in enumerate(val_loader):
                torchvision.utils.save_image(results[idx].unsqueeze(dim=0), pth + '/{}_{}'.format(ep+1,name[0]))
        return all_psnr/len(val_loader)

def train(val_loader,model,lf_1,lf_2,opt,scheduler,scheduler2,args):
    tv_loss = TVLoss()
    lf_mse = nn.MSELoss()
    pth=args.save_pth
    max_psnr=0
    model=model.train()
    if os.path.exists(pth):
        shutil.rmtree(pth)
    os.makedirs(pth)
    transforms_imag = torchvision.transforms.Compose([
        transforms.RandomCrop([256, 256], padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor()])
    with tqdm(total=args.epochs) as tq:
        for ep in range(args.epochs):
            running_loss=0.0
            dataset_train = Dataset_pair(args.input_root, args.label_root, transform=transforms_imag, noise_root=args.noise_root)
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True,
                                                       pin_memory=True,
                                                       drop_last=False,num_workers=8)
            with tqdm(total=len(train_loader)) as tq2:
                for train_batch,(data,gt,noise_gt) in enumerate(train_loader):
                    tr_patch_h, tr_patch_w, tr_step_h, tr_step_w=get_size(data.shape,args.num_of_patch)
                    if torch.cuda.is_available():
                        data=data.cuda()
                        gt=gt.cuda()
                        noise_gt=noise_gt.cuda()
                    gt_after,_,_,_=crop_img(gt, tr_patch_h, tr_patch_w, tr_step_h, tr_step_w, args.batchsize)
                    pred,att_map,noise_map,pred_tem=model(data)
                    loss = (20 * (lf_1(pred, gt) + 1 - lf_2(pred, gt)
                                      + lf_1(vgg19(pred), vgg19(gt)) / (
                                                  pred.shape[-1] * pred.shape[-2] * pred.shape[-3])
                                      )
                                + 15 * lf_attention(att_map, gt, data)
                                + 15 * lf_noise(noise_map, noise_gt, data)
                                + 50 * lf_mse(pred_tem[1],gt_after)
                                + 50 * color_loss(pred, gt)
                                + 30 * tv_loss(pred))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    running_loss += loss.item()
                    tq2.set_description(desc="epoch:{}".format(ep+1),refresh=False)
                    tq2.update(1)
                    # for name, parms in model.named_parameters():
                    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                    #           ' -->grad_value:', parms.grad)
            now_loss = running_loss / len(train_loader)
            scheduler.step(metrics=now_loss)
            scheduler2.step()
            output_infos = '\rTrain===> [epoch {}/{}] [loss {:.3f}] '.format(
                ep+1, args.epochs, now_loss)
            print(output_infos)
            val_psnr=val(val_loader,model,ep,pth,max_psnr,args)
            if val_psnr>max_psnr :
                max_psnr=val_psnr
                if val_psnr>args.model_save_psnr_st:
                    torch.save(model.state_dict(), pth + '/{}_psnr:{}.pth'.format(ep + 1, val_psnr))
            if (ep == args.epochs - 1):
                torch.save(model.state_dict(), pth + '/{}_psnr:{}.pth'.format(ep + 1, val_psnr))
            tq.set_description(desc="val:{}".format(val_psnr),refresh=False)
            tq.update(1)



if __name__=='__main__':
    args = parser.parse_args()
    dataset_val=Dataset_pair(args.val_root,args.val_gt_root,train=False,transform=torchvision.transforms.ToTensor())
    val_loader=torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, drop_last=False)
    mbe=IEDCN()
    if torch.cuda.is_available():
        mbe = mbe.cuda()
        # mbe=nn.DataParallel(mbe,device_ids)
    lf_1=nn.L1Loss()
    lf_2=SSIM()
    opt=torch.optim.Adam(mbe.parameters(),lr=args.lr,betas=(0.9,0.999),eps=1e-8)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=list(range(0, args.epochs, 1)), gamma=0.98)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=False,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    train(val_loader, mbe, lf_1, lf_2, opt, scheduler, scheduler2,args)