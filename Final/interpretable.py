'''
Modified Date: 2022/01/11
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import os
import tqdm
from dataset import food_data
from torch.utils.data import DataLoader
import argparse
from model.model import resnest50
import torch
import numpy as np
import random
from utils import grad_cam
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_layer(opt, model, test_loader):
    model.eval()
    for image, label, file_path in tqdm.tqdm(test_loader):
        label = label.numpy()[0]
        if label == 687:
            image = image.cuda()
            pred = model(image) #(B, num_classes)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            grayscale_cam = grad_cam(image, model)
            for i, gray_cam in enumerate(grayscale_cam):
                file_path_ = file_path[i]
                image_bgr = cv2.imread(os.path.join(opt.data_path, 'val', str(label), file_path_))
                image_bgr = cv2.resize(image_bgr, (400,400))
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb = (image_rgb / 255.0).astype(np.float32)
                visualization = show_cam_on_image(image_rgb, gray_cam, use_rgb=True)
                plt.imshow(visualization)
                print(pred_label[i], label)
                plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../dataset', help='path to data')
    parser.add_argument('--csv_path', default='../../dataset/testcase', help='path to testing csv')
    parser.add_argument('--output_path', default='./pred', help='path to testing csv')
    parser.add_argument('--semi', action = 'store_true', help='Whether to generate psuedo label')

    parser.add_argument('--model', default='resnest50', help='resnet50/resnet101/resnest50/resnest101')
    parser.add_argument('--aug_ensemble', action = 'store_true', help='Whether to use five_crop')
    parser.add_argument('--val', action = 'store_true', help='validation mode or testing mode')
    parser.add_argument('--model_ensemble', action = 'store_true', help='Whether to ensemble')
    parser.add_argument('--load', default='./checkpoints/resnest50/model_epoch4_all79.49_f89.12_c75.60_r36.25.pth', help='path to model to continue training')

    parser.add_argument('--batch_size', type=int, default=1, help='number of cpu workers')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu workers')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    random_seed_general = 500
    random.seed(random_seed_general) 
    torch.manual_seed(random_seed_general)
    torch.cuda.manual_seed_all(random_seed_general)
    np.random.seed(random_seed_general)
    random.seed(random_seed_general)
    torch.backends.cudnn.deterministic = True
    # prepare_images(opt.data_path, dcm_path, png_path+'/', opt.train_val_ratio, opt.sub_folder)
    
    model = resnest50(True, 1000).cuda()
    model.load_state_dict(torch.load(opt.load))
    

    test_data = food_data(opt.data_path, 'val', augment_ensemble=opt.aug_ensemble)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, pin_memory=True)
    visualize_layer(opt, model, test_loader)
    