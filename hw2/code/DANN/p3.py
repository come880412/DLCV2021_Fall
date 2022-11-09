import torch
from dataset import DANN_data
from model import DANN
from torch.utils.data import DataLoader
import sys
import numpy as np
import csv

import os

def test(model, pred_path, data_path):
    model.eval()
    csv_out = open(pred_path, 'w')
    csv_writer = csv.writer(csv_out, delimiter=',')
    csv_writer.writerow(['image_name', 'label'])
    data_test = DANN_data(data_path, None, mode='savecsv')
    dataloader = DataLoader(data_test, batch_size=128, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch, (img, image_name_list) in enumerate(dataloader):
            img = torch.FloatTensor(img)
            img = img.cuda()

            class_out = model(img)
            pred = class_out.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            for i in range(len(pred_label)):
                image_name = image_name_list[i]
                csv_writer.writerow([image_name, pred_label[i]])
    
    
if __name__ == '__main__':

    data_path = sys.argv[1]
    target = sys.argv[2]
    pred_path = sys.argv[3]

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = DANN().cuda()
    model.load_state_dict(torch.load('./model/p3/%s/DANN.pth' % (target)) )

    test(model, pred_path, data_path)