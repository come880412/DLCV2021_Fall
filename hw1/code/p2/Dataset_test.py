from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
Class = ['Urban','Agriculture','Rangeland','Forest','Water','Barren','Unknown']
Colormap = [[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]
Colormap_dict = {(0,255,255):0, (255,255,0):1, (255,0,255):2, (0,255,0):3, (0,0,255):4, (255,255,255):5, (0,0,0):6, (255,0,0):2,
                 0:(0,255,255), 1:(255,255,0), 2:(255,0,255), 3:(0,255,0), 4:(0,0,255), 5:(255,255,255), 6:(0,0,0)}
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.40851283, 0.37851363, 0.28088605), (0.14234419, 0.10848415, 0.09824672))
])

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

    
