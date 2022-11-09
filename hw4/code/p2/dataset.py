from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import csv
import numpy as np

train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

val_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

class miniImagenet(Dataset):
    def __init__(self, root):
        self.image_path_list = []
        image_name_list = os.listdir(root)
        image_name_list.sort()

        for image_name in image_name_list:
            image_path = os.path.join(root, image_name)
            self.image_path_list.append(image_path)
    
    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path).convert('RGB')
        image = val_transform(image)

        return image
    
    def __len__(self):
        return len(self.image_path_list)

class office_data(Dataset):
    def __init__(self, root, label_path, mode='train'):
        self.image_path_list = []
        self.mode = mode
        label_dict = {}
        with open(label_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                label_dict[row[0]] = row[1]
        
        # Extract label dictionary
        # val_csv_path = os.path.join(root, 'val.csv')
        # csv_data = np.loadtxt(val_csv_path, delimiter=',', dtype=np.str)[1:]
        # label_count = 0
        # for data in csv_data:
        #     label = data[2]
        #     if label not in label_dict.keys():
        #         label_dict[label] = label_count
        #         label_dict[label_count] = label
        #         label_count += 1
        # with open('./label.csv', 'w') as csv_file:
        #     writer = csv.writer(csv_file)
        #     for key, value in label_dict.items():
        #         writer.writerow([key, value])

        # Extract data
        if self.mode == 'train':
            train_csv_path = os.path.join(root, 'train.csv')
            csv_data = np.loadtxt(train_csv_path, delimiter=',', dtype=np.str)[1:]
            self.transform = train_transform
        elif self.mode == 'val':
            val_csv_path = os.path.join(root, 'val.csv')
            csv_data = np.loadtxt(val_csv_path, delimiter=',', dtype=np.str)[1:]
            self.transform = val_transform
        
        for data in csv_data:
            file_name = data[1]
            label_name = data[2]
            label = int(label_dict[label_name])
            file_path = os.path.join(root, self.mode, file_name)
            self.image_path_list.append([file_path, label])

    def __getitem__(self, index):
        image_path, label = self.image_path_list[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, label
    
    def __len__(self):
        return len(self.image_path_list)

class predict_data(Dataset):
    def __init__(self, test_img_csv, test_images_dir):
        self.image_path_list = []
        
        # Extract data
        csv_data = np.loadtxt(test_img_csv, delimiter=',', dtype=np.str)[1:]
        self.transform = val_transform
        
        for data in csv_data:
            file_name = data[1]
            file_path = os.path.join(test_images_dir, file_name)
            self.image_path_list.append(file_path)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image
    
    def __len__(self):
        return len(self.image_path_list)