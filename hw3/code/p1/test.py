import numpy as np
from dataset import p1_data
from torch.utils.data import DataLoader
import torch.nn as nn
from timm import create_model
import torch
import sys

def test(model ,test_loader, csv_path):
    cuda = True if torch.cuda.is_available() else False
    model.eval()
    if cuda:
        model = model.cuda()
    csv_save = [['image_id', 'label']]
    for image, image_name in test_loader:
        with torch.no_grad():
            if cuda:
                image = image.cuda()
            pred = model(image)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            for i in range(len(image_name)):
                csv_save.append([image_name[i], pred_label[i]])
    np.savetxt(csv_path, csv_save, encoding='utf-8-sig', fmt='%s', delimiter=',')

if __name__ == '__main__':
    test_data_path = sys.argv[1]
    csv_path = sys.argv[2]
    saved_model = 'model_epoch8_acc95.73.pth'
    
    test_data = p1_data(test_data_path, 'test')
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=8)
    model = create_model('vit_large_patch16_224', pretrained=True)
    model.head = nn.Linear(in_features= model.head.in_features, out_features=37, bias=True)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(saved_model))
    test(model, test_loader, csv_path)