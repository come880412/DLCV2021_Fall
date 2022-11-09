from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
 ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
class GAN_data(Dataset):
    def __init__(self,root,mode):
        self.imgfile = []
        if mode == 'train':
            self.transform = transform_train
        else:
            self.transform = transform_test
        root_dir = os.listdir(root)
        root_dir.sort()
        for image_name in root_dir:
            img_path = os.path.join(root, image_name)
            self.imgfile.append(img_path)
    
    def __getitem__(self,index):
        image = self.imgfile[index]
        image = Image.open(image)
        image = self.transform(image)
        return image
        
    def __len__(self):
        return len(self.imgfile)