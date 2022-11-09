import numpy as np
from dataset import p1_data
from torch.utils.data import DataLoader
import torch.nn as nn
from timm import create_model
import torch
import sys

def test(csv_path):
    csv_load = np.loadtxt(csv_path, delimiter=',', dtype=np.str)[1:]
    correct = 0
    total = 0
    for i in csv_load:
        img_name = i[0]
        pred_label = int(i[1])
        label = int(img_name.split('_')[0])
        if pred_label == label:
            correct += 1
        total += 1
    acc = round((correct / total)* 100, 2)
    print('acc:', acc)

if __name__ == '__main__':
    csv_path = './output.csv'
    
    test(csv_path)