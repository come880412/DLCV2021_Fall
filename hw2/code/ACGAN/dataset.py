from torch.utils.data import Dataset
import numpy as np
import os
import PIL.Image as Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])

class ACGAN_data(Dataset):
    def __init__(self, root, mode):
        self.image_list = []
        path = os.path.join(root, mode)
        with open(path + '.csv', encoding='utf-8') as f:
            data = f.read().splitlines()
            for idx, line in enumerate(data):
                if idx == 0:
                    continue
                image_label_list = line.split(',')
                image_path = os.path.join(path, image_label_list[0])
                label = int(image_label_list[1])
                self.image_list.append([image_path, label])
    def __getitem__(self, index):
        filename, label = self.image_list[index]
        image = Image.open(filename).convert('RGB')
        image = transform(image)
        return [image, label]

    def __len__(self):
        return len(self.image_list)

class cls_data(Dataset):
    def __init__(self, root):
        self.image_pair = []
        image_list = os.listdir(root)
        image_list.sort()
        for image_name in image_list:
            label = image_name.split('_')[0]
            image_path = os.path.join(root, image_name)
            self.image_pair.append([image_path, int(label)])
    def __getitem__(self, index):
        image_path, label = self.image_pair[index]
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        return image, label

    def __len__(self):
        return len(self.image_pair)