import torch
import numpy as np
import argparse
import torch.nn as nn
import os
from Dataset import val_seg
from VGG16_FCN32s import VGG16_FCN32s
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import tqdm
from utils import pred_to_mask, mean_iou_evaluate
import cv2

def validation(args, model, val_loader):

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        for image, label, image_name in tqdm.tqdm(val_loader):
            image, label = image.cuda(), label.type(torch.LongTensor).cuda()
            pred = model(image)
            if args.model == 'deeplabv3_resnet50':
                pred = pred['out']
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            segment_image = pred_to_mask(pred_label)
            for batch in range(len(segment_image)):
                output_image = np.array(segment_image[batch]).astype(np.uint8)
                image_save_name = '%s/%s' % (args.pred_path, image_name[batch])
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_save_name, output_image)
    mIoU_cal = mean_iou_evaluate()
    mIoU = mIoU_cal.calculate(args.pred_path, args.root_val)
    mIoU *= 100
    print('validation mIoU:', mIoU, '%')
    return mIoU
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_val", help="validation image path",default='../../hw1_data/p2_data/validation/')
    parser.add_argument('--model', default='deeplabv3_resnet50', help='VGG16_FCN32s/deeplabv3_resnet50')
    parser.add_argument('--saved_model', default='../../saved_models/p2/deeplabv3_resnet50/model_miou71.54.pth', help='path to model to continue training')
    parser.add_argument("--gpu_id", help="gpu_id",default=0)
    parser.add_argument('--pred_path', default='./pred_deeplabv3_resnet50', help='path to save seg iamges')
    args = parser.parse_args()
    os.makedirs(args.pred_path, exist_ok=True)
    val_data = val_seg(args.root_val)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=8)

    if args.saved_model:
        if args.model == 'VGG16_FCN32s':
            model = VGG16_FCN32s(num_classes=7, pretrained=False)
        elif args.model == 'deeplabv3_resnet50':
            model = deeplabv3_resnet50(pretrained=False, num_classes=7, aux_loss=True)
        print('load pretrained model!')
        model.load_state_dict(torch.load(args.saved_model))
    
    validation(args,model, val_loader)