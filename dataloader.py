from torch.utils.data import Dataset
import os
import random
import numpy as np
from PIL import Image
from PIL import ImageEnhance
class Dataset_pair(Dataset):
    def __init__(self, input_root, label_root, train=True,transform=None,noise_root=None):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        if noise_root!=None:
            self.noise_root = noise_root
            self.noise_files = os.listdir(noise_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.train=train
        self.transforms = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = Image.open(input_img_path)

        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = Image.open(label_img_path)

        if self.train:
            noise_img_path = os.path.join(self.noise_root, self.noise_files[index])
            noise_img = Image.open(noise_img_path)
            enh_con = ImageEnhance.Contrast(label_img)
            contrast = 1.35
            label_img= enh_con.enhance(contrast)
            seed = np.random.randint(17)
            if self.transforms:
                random.seed(seed)
                input_img = self.transforms(input_img)
                random.seed(seed)
                label_img = self.transforms(label_img)
                random.seed(seed)
                noise_img = self.transforms(noise_img)
            return (input_img, label_img,noise_img)
        else:
            seed = np.random.randint(17)
            if self.transforms:
                random.seed(seed)
                input_img = self.transforms(input_img)
                random.seed(seed)
                label_img = self.transforms(label_img)
            return (input_img, label_img,self.input_files[index])

class Dataset_no_pair(Dataset):
    def __init__(self, input_root,transform=None):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.transforms = transform

    def __len__(self):
        return len(self.input_files)
    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = Image.open(input_img_path)

        seed = np.random.randint(17)
        if self.transforms:
            random.seed(seed)
            input_img = self.transforms(input_img)
            random.seed(seed)
        return (input_img,self.input_files[index])