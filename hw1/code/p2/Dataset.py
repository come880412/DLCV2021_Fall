from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
from utils import mask_to_label
Class = ['Urban','Agriculture','Rangeland','Forest','Water','Barren','Unknown']
Colormap = [[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]
Colormap_dict = {(0,255,255):0, (255,255,0):1, (255,0,255):2, (0,255,0):3, (0,0,255):4, (255,255,255):5, (0,0,0):6, (255,0,0):2,
                 0:(0,255,255), 1:(255,255,0), 2:(255,0,255), 3:(0,255,0), 4:(0,0,255), 5:(255,255,255), 6:(0,0,0)}
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.40851283, 0.37851363, 0.28088605), (0.14234419, 0.10848415, 0.09824672))
])
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
class img_seg(Dataset):
    def __init__(self,root, crop_size):
        self.imgfile = []
        self.labelfile = []
        self.crop_size = crop_size
        path_root = os.listdir(root)
        path_root.sort()
        for image_name in path_root:
            root_img = os.path.join(root, image_name)
            if image_name.split('_')[1] == 'sat.jpg':
                self.imgfile.append(root_img)
            else:
                self.labelfile.append(root_img)
        # self.imgfile = self.imgfile[:10]
        # self.labelfile = self.labelfile[:10]
        self.augment_rotation_param = np.random.randint(0, 4, len(self.imgfile))
        self.augment_flip_param = np.random.randint(0, 3, len(self.imgfile))
        if self.crop_size:
            self.random_crop_paramx = np.random.randint(0, 512-self.crop_size, len(self.imgfile))
            self.random_crop_paramy = np.random.randint(0, 512-self.crop_size, len(self.imgfile))
    def __getitem__(self,index):
        image_path = self.imgfile[index]
        image_name = image_path.split('/')[-1]
        label_path = self.labelfile[index]
        image = cv2.imread(image_path)                                                                                                                                                                             
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_img = cv2.imread(label_path)                                                                                                                                                                      
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        label_img = label_img.transpose((2, 0, 1))
        if self.crop_size:
            label_img = label_img[:, self.random_crop_paramx[index]:self.random_crop_paramx[index]+self.crop_size, 
                                            self.random_crop_paramy[index]:self.random_crop_paramy[index]+self.crop_size]
            image = image[:, self.random_crop_paramx[index]:self.random_crop_paramx[index]+self.crop_size, 
                                    self.random_crop_paramy[index]:self.random_crop_paramy[index]+self.crop_size]
        if not self.augment_flip_param[index] == 0:
            image = np.flip(image, self.augment_flip_param[index])
            label_img = np.flip(label_img, self.augment_flip_param[index])
        if not self.augment_rotation_param[index] == 0:
            image = np.rot90(image, self.augment_rotation_param[index], (1, 2))
            label_img = np.rot90(label_img, self.augment_rotation_param[index], (1, 2))
        label = mask_to_label(label_img)
        """Visualization"""
        # img = image.transpose((1, 2, 0))
        # img = img[250:260,250:260,:]
        # label_img_ = label_img.transpose((1, 2, 0))
        # label_img_ = label_img_[250:260,250:260,:]
        # plt.subplot(121)
        # plt.imshow(img)
        # plt.subplot(122)
        # plt.imshow(label_img_)
        # print(label[250:260,250:260])
        # plt.show()
        
        img = torch.from_numpy((image.copy()))
        img = img.float()/255.0
        mean = torch.as_tensor([0.40851283, 0.37851363, 0.28088605], dtype=img.dtype, device=img.device)
        std = torch.as_tensor([0.14234419, 0.10848415, 0.09824672], dtype=img.dtype, device=img.device)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        img.sub_(mean).div_(std)
        return img, label, image_name

    def __len__(self):
        return len(self.imgfile)
class val_seg(Dataset):
    def __init__(self, root):
        self.imgfile = []
        self.labelfile = []
        path_root = os.listdir(root)
        path_root.sort()
        for image_name in path_root:
            root_img = os.path.join(root, image_name)
            if image_name.split('_')[1] == 'sat.jpg':
                self.imgfile.append(root_img)
            else:
                self.labelfile.append(root_img)
        
    def __getitem__(self,index):
        image_path = self.imgfile[index]
        label_path = self.labelfile[index]
        image_name = label_path.split('/')[-1]
        label_img = cv2.imread(label_path)
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        label_img = label_img.transpose((2, 0, 1))
        label = mask_to_label(label_img)
        image = Image.open(image_path)
        image = transform(image)
        return image, label, image_name

    def __len__(self):
        return len(self.imgfile)

class test_seg(Dataset):
    def __init__(self, root):
        self.imgfile = []
        self.labelfile = []
        path_root = os.listdir(root)
        path_root.sort()
        for image_name in path_root:
            root_img = os.path.join(root, image_name)
            self.imgfile.append(root_img)
        
    def __getitem__(self,index):
        image_path = self.imgfile[index]
        image_name = image_path.split('/')[-1]
        image = Image.open(image_path)
        image = transform(image)
        return image, image_name

    def __len__(self):
        return len(self.imgfile)

if __name__ == '__main__':
    pass
    # dataset = img_seg('../../../hw1_data/p2_data/train/', 'train')
    # train_loader = DataLoader(dataset,batch_size=1,shuffle=True, num_workers=0)
    # iter_train = iter(train_loader)
    # image, label, image_name = iter_train.next()
    # dataset = img_seg('../../../hw1_data/p2_data/validation/')
    # val_loader = DataLoader(dataset,batch_size=1,shuffle=False, num_workers=0)
    # iter_val = iter(val_loader)
    # image, label, image_name = iter_val.next()

    
