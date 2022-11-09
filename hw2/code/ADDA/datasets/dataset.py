from torch.utils.data import Dataset
import numpy as np
import os
import PIL.Image as Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])

class ADDA_data(Dataset):
    def __init__(self, data_root, csv_root, mode):
        self.image_list = []
        self.mode = mode
        if mode == 'val' or mode == 'train':
            path = os.path.join(data_root, 'train')
            csv_file = os.path.join(csv_root, mode + '.csv')
            with open(csv_file, encoding='utf-8') as f:
                data = f.read().splitlines()
                for idx, line in enumerate(data):
                    image_label_list = line.split(',')
                    image_path = os.path.join(path, image_label_list[0])
                    label = int(image_label_list[1])
                    self.image_list.append([image_path, label])
        elif mode == 'test':
            path = data_root
            csv_file = os.path.join(csv_root, mode + '.csv')
            with open(csv_file, encoding='utf-8') as f:
                data = f.read().splitlines()
                for idx, line in enumerate(data):
                    if idx == 0:
                        continue
                    image_label_list = line.split(',')
                    image_path = os.path.join(path, image_label_list[0])
                    label = int(image_label_list[1])
                    self.image_list.append([image_path, label])
        elif mode == 'savecsv':
            image_list = os.listdir(data_root)
            image_list.sort()
            for image_name in image_list:
                image_path = os.path.join(data_root, image_name)
                self.image_list.append([image_path, image_name])

    def __getitem__(self, index):
        if self.mode == 'savecsv':
            file_path, file_name = self.image_list[index]
            image = Image.open(file_path).convert('RGB')
            image = transform(image)
            return [image, file_name]
        else:
            filename, label = self.image_list[index]
            image = Image.open(filename).convert('RGB')
            image = transform(image)
            return [image, label]

    def __len__(self):
        return len(self.image_list)
