import torch
from byol_pytorch import BYOL
from torch.nn.modules.batchnorm import BatchNorm1d
from torchvision import models
from dataset import office_data, predict_data
import argparse
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import csv
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

def csv_compare(args):
    test_data = np.loadtxt(args.csv_path, delimiter=',', dtype=np.str)[1:]
    output_data = np.loadtxt(args.output_csv, delimiter=',', dtype=np.str)[1:]
    correct = 0
    total = len(test_data)
    for idx, data in enumerate(test_data):
        label = data[2]
        pred = output_data[idx][2]
        if label == pred:
            correct += 1
    print('test accuracy: %.2f%%' % ((correct / total)*100))
        
def predict(args):
    label_dict = {}
    with open(args.label_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            label_dict[row[0]] = row[1]

    fc_layer = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(True),
        nn.Linear(512, 65),
    )

    val_data = predict_data(args.csv_path, args.data_dir)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)

    resnet = models.resnet50(pretrained=False)
    
    if args.load:
        resnet.fc = fc_layer
        resnet.load_state_dict(torch.load(args.load))
    
    resnet = resnet.cuda()

    csv_data = np.loadtxt(args.csv_path, delimiter=',', dtype=np.str)
    # Validation
    resnet.eval()
    pred_count = 1
    with torch.no_grad():
        for images in val_loader:
            images = images.cuda()
            pred = resnet(images)

            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            for pred_ in pred_label:
                label_name = label_dict[str(pred_)]
                csv_data[pred_count][2] = label_name
                pred_count += 1
    np.savetxt(args.output_csv, csv_data, fmt='%s', delimiter=',')
    # csv_compare(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--load', type=str, default='./model_p2.pth', help="Model checkpoint path")
    parser.add_argument('--label_path', default='./code/p2/label.csv', type=str, help="Training images directory")
    parser.add_argument('csv_path', default='../../hw4_data/office/val.csv', type=str, help="Training images directory")
    parser.add_argument('data_dir', default='../../hw4_data/office/val', type=str, help="Training images directory")
    parser.add_argument('output_csv', default='./output.csv', type=str, help="Training images directory")
    args = parser.parse_args()

    predict(args)
    

    