from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import numpy as np

transforms_test =    transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5)),])

transforms_train =  transforms.Compose([

                    transforms.RandomResizedCrop((224, 224)),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.GaussianBlur(7, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5)),
                    ])
class p1_data(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.image_name = []
        self.filename = []
        self.mode = mode
        if mode == 'train':
            self.transform = transforms_train
        elif mode == 'test' or mode == 'val':
            self.transform = transforms_test
        
        #Load data
        image_name_list = os.listdir(self.root)
        image_name_list.sort()

        for image_name in image_name_list:
            self.image_name.append(image_name)
            filepath = os.path.join(self.root,image_name)
            if self.mode == 'train' or self.mode == 'val':
                label = int(image_name.split('_')[0])
                self.filename.append([filepath, label])
            else:
                self.filename.append(filepath)

    def __getitem__(self,index):
        if self.mode == 'train' or self.mode == 'val':
            image_path, label = self.filename[index]
            image_name = self.image_name[index]
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image, label, image_name
        elif self.mode == 'test':
            image_path = self.filename[index]
            image_name = self.image_name[index]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform is not None:
                image = self.transform(image)
            return image, image_name
        
    def __len__(self):
        return len(self.filename)