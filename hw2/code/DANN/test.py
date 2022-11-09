import torch
import numpy as np
import argparse
import logging
from dataset import DANN_data
import time
from model import DANN
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
from torch.optim import Adam

def test(model, args):
    model.eval()
    correct = 0
    data_test = DANN_data('../../hw2_data/digits/%s'%args.target, '../../hw2_data/digits/%s'%args.target, mode='test')
    dataloader = DataLoader(data_test, batch_size=128, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch, (img, label) in enumerate(dataloader):
            img = torch.FloatTensor(img)
            class_label = label.long()
            img = img.cuda()
            class_label = class_label.cuda()

            class_out = model(img)
            pred = class_out.max(1, keepdim=True)[1]
            correct += pred.eq(class_label.view_as(pred)).sum().item()
    ACC = correct/len(data_test)
    print('target domain ACC:%.4f'% ACC)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='DANN implement')
    parser.add_argument('--model_path',type=str,default='./checkpoints/transfer/usps/model_epoch94_usps_svhn_sourceacc98.15_domainacc69.38.pth')
    parser.add_argument('--source',type=str,default= 'usps') # mnistm, svhn, usps
    parser.add_argument('--target',type=str,default='svhn') # usps, mnistm, svhn 
    parser.add_argument('--GPUID',type=str,default= '0')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPUID
    model = DANN().cuda()
    model.load_state_dict(torch.load(args.model_path))

    test(model, args)