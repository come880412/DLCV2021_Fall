from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import center_crop

def transform(image_size, mode='train'):
    transforms_train =  transforms.Compose([
                            transforms.Resize((512, 512)),
                            # transforms.RandomResizedCrop(400),
                            transforms.CenterCrop(image_size),
                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                            transforms.RandomRotation(30),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.637, 0.545, 0.426), (0.280, 0.287, 0.312)),
                        ])

    transforms_test =   transforms.Compose([
                            transforms.Resize((512, 512)),
                            transforms.CenterCrop(image_size),
                            # transforms.Resize((448, 448)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.637, 0.545, 0.426), (0.280, 0.287, 0.312)),])
    if mode == 'train' or mode=='semi':
        return transforms_train
    elif mode == 'val' or mode == 'test':
        return transforms_test

def test_aug(image_tensor, ori_image, image_size):
    five_crop = transforms.Compose([
        transforms.Resize((512, 512)),
    transforms.FiveCrop(256), # this is a list of PIL Images
    ])

    five_transform = transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.637, 0.545, 0.426), (0.280, 0.287, 0.312)),])


    rotation_augment = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(image_size),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.637, 0.545, 0.426), (0.280, 0.287, 0.312)),
    ])

    colorjitter_augment = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(image_size),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        transforms.Normalize((0.637, 0.545, 0.426), (0.280, 0.287, 0.312)),
    ])

    randomhorizontalFlip_augment = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(1),
        transforms.ToTensor(),
        transforms.Normalize((0.637, 0.545, 0.426), (0.280, 0.287, 0.312)),
    ])

    image_five = five_crop(ori_image)
    for image_pil in image_five:
        image_tensor = torch.cat((image_tensor, five_transform(image_pil).unsqueeze(0)), dim=0)
    image_tensor = torch.cat((image_tensor, rotation_augment(ori_image).unsqueeze(0)), dim=0)
    image_tensor = torch.cat((image_tensor, randomhorizontalFlip_augment(ori_image).unsqueeze(0)), dim=0)
    image_tensor = torch.cat((image_tensor, colorjitter_augment(ori_image).unsqueeze(0)), dim=0)
    return image_tensor

def five_show(img_tensor):
    columns=2 # 两列
    rows= math.ceil(5/2) # 计算多少行
    # 把每个 tensor ([c,h,w]) 转换为 image
    for i in range(5):
        img = img_tensor[i]
        plt.subplot(rows, columns, i+1)
        plt.imshow(img)
    plt.show()

class food_data(Dataset):
    def __init__(self, root, mode, image_size, augment_ensemble=None):
        self.image_path = []
        self.image_name = []

        self.mode = mode
        self.root = root
        self.image_size = image_size
        self.label = []
        self.augment_ensemble = augment_ensemble
        if mode == 'train':
            self.transform = transform(image_size, 'train')
            data_path = os.path.join(root, mode)
            data_list = os.listdir(data_path)

            for data in data_list:
                label = int(data)
                image_folder_path = os.path.join(data_path, str(data))
                image_folder_list = os.listdir(image_folder_path)
                random.shuffle(image_folder_list)
                
                for idx, image_name in enumerate(image_folder_list):
                    image_path = os.path.join(image_folder_path, image_name)
                    self.image_path.append([image_path, label])
                    self.image_name.append(image_name)
                    self.label.append(label)
                    
        elif mode == 'val':
            self.transform = transform(image_size, 'val')
            data_path = os.path.join(root, mode)
            data_list = os.listdir(data_path)

            for data in data_list:
                label = int(data)
                image_folder_path = os.path.join(data_path, str(data))
                image_folder_list = os.listdir(image_folder_path)
                
                for idx, image_name in enumerate(image_folder_list):
                    image_path = os.path.join(image_folder_path, image_name)
                    self.image_path.append([image_path, label])
                    self.label.append(label)
                    self.image_name.append(image_name)
        
        elif mode == 'test':
            data_path = os.path.join(root, mode)
            data_list = os.listdir(data_path)
            data_list.sort()

            for image_name in data_list:
                image_path = os.path.join(data_path, image_name)
                self.image_path.append([image_path, image_name.split('.')[0]])
            self.transform = transform(image_size, 'test')
        
        elif mode == 'semi':
            semi_data = np.loadtxt(os.path.join(root, 'val_semi.csv'), delimiter=',', dtype=np.str)
            np.random.shuffle(semi_data)

            self.transform = transform(image_size, 'semi')
            train_path = os.path.join(root, 'train_refined')
            data_list = os.listdir(train_path)
            
            for data in semi_data:
                label = int(data[1])
                image_name = data[0]

                image_path = os.path.join(root, 'val', data[1], image_name)
                self.image_path.append([image_path, label])
                self.image_name.append(image_name)
                self.label.append(label)

            for i in range(1000):
                label = i
                image_folder_path = os.path.join(train_path, str(i))
                image_folder_list = os.listdir(image_folder_path)
                random.shuffle(image_folder_list)

                for idx, image_name in enumerate(image_folder_list):
                    image_path = os.path.join(image_folder_path, image_name)
                    self.image_path.append([image_path, label])
                    self.image_name.append(image_name)
                    self.label.append(label)

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'val' or self.mode == 'semi':
            image_path, label = self.image_path[index]
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, label, self.image_name[index]
        
        elif self.mode == 'test':
            image_path, image_name = self.image_path[index]
            image = Image.open(image_path).convert('RGB')
            if self.augment_ensemble:
                image_tensor = self.transform(image).unsqueeze(0)
                image_tensor = test_aug(image_tensor, image, self.image_size)

                return image_tensor, image_name
            else:
                return self.transform(image), image_name

    def __len__(self):
        return len(self.image_path)
    
    def get_labels(self):
        return self.label
                