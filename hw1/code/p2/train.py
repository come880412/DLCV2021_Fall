from urllib.parse import _NetlocResultMixinBase
import torch
import numpy as np
import argparse
import os

from Dataset import img_seg, val_seg
from VGG16_FCN32s import VGG16_FCN32s
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, RMSprop
import tqdm
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau
import cv2
from utils import pred_to_mask, mean_iou_evaluate, FocalLoss2d, dice_loss, multiclass_dice_coeff
from tensorboardX import SummaryWriter
# from torchsummary import summary

def train(args,model, train_loader, val_loader):
    writer = SummaryWriter('runs/%s' % (args.model))
    cuda = True if torch.cuda.is_available() else False
    if args.loss == 'focal':
        print('use focal loss!')
        criterion = FocalLoss2d(gamma=2)
    elif args.loss == 'crossentropy':
        print('use crossentropy loss!')
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'BCE':
        print('use BCEwithlogits loss')
        criterion = nn.BCEWithLogitsLoss()
        
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay = 5e-4)
    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    """lr_scheduler"""
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - args.lr_decay_epoch) / float(args.lr_decay_epoch + 1)
        return lr_l
    if args.scheduler == 'linear':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=0.412)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5) # goal: maximize Dice score

    print('Start training!!')
    train_update = 0
    max_miou = 0
    for epoch in range(args.n_epochs):
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        loss_total = 0.
        label_total = 0
        correct_total = 0
        acc = 0.
        for image, label, _ in train_loader:
            image = image.cuda()
            label = label.type(torch.LongTensor).cuda()
            optimizer.zero_grad()
            pred = model(image) # (batch, 7, 512, 512)
            if args.model == 'deeplabv3_resnet50' or args.model == 'deeplabv3_resnet101':
                pred_out = pred['out']
                pred_aux = pred['aux']
                if args.dice_loss:
                    loss = criterion(pred_out, label) + criterion(pred_aux, label) \
                            + dice_loss(F.softmax(pred_out, dim=1).float(),
                                        F.one_hot(label, 7).permute(0, 3, 1, 2).float(),
                                        multiclass=True)  \
                            + dice_loss(F.softmax(pred_aux, dim=1).float(),
                                      F.one_hot(label, 7).permute(0, 3, 1, 2).float(),
                                      multiclass=True)
                else:
                    loss = criterion(pred_out, label) + criterion(pred_aux, label)
                pred = pred['out']
            else:
                if args.dice_loss:
                    loss = criterion(pred, label) \
                            + dice_loss(F.softmax(pred, dim=1).float(),
                                        F.one_hot(label, 7).permute(0, 3, 1, 2).float(),
                                        multiclass=True)
                else:
                    loss = criterion(pred, label)
            loss_total += loss.item()
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += (pred_label.shape[0] * pred_label.shape[1] * pred_label.shape[2]) # batch * img_size
            correct_total += correct
            acc = (correct_total / label_total) * 100
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_total:.4f}",
            Accuracy=f"{acc:.2f}%"
            )
            writer.add_scalar('training loss', loss, train_update)
            writer.add_scalar('training accuracy', acc, train_update)
            train_update += 1
        pbar.close()
        miou, val_loss = validation(args, model, val_loader)
        if max_miou <= miou:
            print('save model!!')
            max_miou = miou
            torch.save(model.state_dict(), '../../saved_models/p2/%s/model_epoch%d_miou%.2f.pth' % (args.model, epoch, max_miou))
        if args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
    print('max miou:', max_miou)

def validation(args, model, val_loader):
    if args.loss == 'focal':
        criterion = FocalLoss2d(gamma=2)
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        criterion = criterion.cuda()
    model.eval()
    with torch.no_grad():
        loss_total = 0.
        correct = 0
        label_total = 0
        correct_total = 0
        pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="val", unit=" step")
        for image, label, image_name in val_loader:
            image, label = image.cuda(), label.type(torch.LongTensor).cuda()
            pred = model(image)
            if args.model == 'deeplabv3_resnet50' or args.model == 'deeplabv3_resnet101':
                pred = pred['out']
            if args.dice_loss:
                mask_label = F.one_hot(label, 7).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(pred.argmax(dim=1), 7).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                loss_total += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_label[:, 1:, ...], reduce_batch_first=False)
            else:
                loss = criterion(pred, label)
                loss_total += loss.item()
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            segment_image = pred_to_mask(pred_label)
            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += (pred_label.shape[0] * pred_label.shape[1] * pred_label.shape[2]) # batch * img_size
            correct_total += correct
            acc = (correct_total / label_total) * 100
            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_total:.4f}",
            Accuracy=f"{acc:.2f}%"
            )
            for batch in range(len(segment_image)):
                output_image = np.array(segment_image[batch]).astype(np.uint8)
                image_save_name = '%s/%s' % (args.pred_path, image_name[batch])
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_save_name, output_image)
    mIoU_cal = mean_iou_evaluate()
    mIoU = mIoU_cal.calculate(args.pred_path, args.root_val)
    mIoU *= 100
    pbar.update()
    pbar.set_postfix(
    loss=f"{loss_total:.4f}",
    Accuracy=f"{acc:.2f}%",
    mIoU=f"{mIoU:.2f}%")
    pbar.close()
    return mIoU, loss_total / len(val_loader)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_train", help="training image path",default='../../hw1_data/p2_data/train/')
    parser.add_argument("--root_val", help="validation image path",default='../../hw1_data/p2_data/validation/')
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--lr_decay_epoch", type=int, default=30, help="Start to decay epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument('--loss', default='focal', help='loss function(crossentropy/focal)')
    parser.add_argument('--dice_loss', type=int, default=1, help='1(use)/0')
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu_id")
    parser.add_argument("--crop_size", type=int, default=448, help="training size")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument('--optimizer', default='sgd', help='adam/sgd/rmsprop')
    parser.add_argument('--model', default='deeplabv3_resnet101', help='VGG16_FCN32s/deeplabv3_resnet50/deeplabv3_resnet101')
    parser.add_argument('--scheduler', default='step', help='linear/step/ReduceLROnPlateau')
    parser.add_argument('--saved_model', default='', help='path to model to continue training')
    parser.add_argument('--pred_path', default='./pred_deeplabv3_resnet101', help='path to save seg iamges')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu workers')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs('../../saved_models/p2/%s' % (args.model), exist_ok=True)
    os.makedirs(args.pred_path, exist_ok=True)

    train_data = img_seg(args.root_train, args.crop_size)
    val_data = val_seg(args.root_val)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)
    if args.saved_model:
        if args.model == 'VGG16_FCN32s':
            model = VGG16_FCN32s(num_classes=7, pretrained=False)
        elif args.model == 'deeplabv3_resnet50':
            model = deeplabv3_resnet50(pretrained=False, num_classes=7, aux_loss=True)
        print('load pretrained model!')
        model.load_state_dict(torch.load(args.saved_model))
    else:
        if args.model == 'VGG16_FCN32s':
            model = VGG16_FCN32s(num_classes=7)
        elif args.model == 'deeplabv3_resnet50':
            model = deeplabv3_resnet50(pretrained=True)
            model.classifier[-1] = nn.Conv2d(256,7,1,1)
            model.aux_classifier[-1] = nn.Conv2d(256,7,1,1)
        elif args.model == 'deeplabv3_resnet101':
            model = deeplabv3_resnet101(pretrained=True)
            model.classifier[-1] = nn.Conv2d(256,7,1,1)
            model.aux_classifier[-1] = nn.Conv2d(256,7,1,1)
    # summary(model, (3, 448, 448))
    train(args, model, train_loader, val_loader)
    
    