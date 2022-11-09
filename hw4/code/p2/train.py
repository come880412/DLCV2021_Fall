import torch
from byol_pytorch import BYOL
from torch.nn.modules.batchnorm import BatchNorm1d
from torchvision import models
from dataset import miniImagenet, office_data
import argparse
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os

import warnings
warnings.filterwarnings("ignore")

def SSL(args):
    train_data = miniImagenet(args.data_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

    resnet = models.resnet50(pretrained=False).cuda()

    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=args.lr)

    print('Start training!')
    for epoch in range(args.n_epochs):
        total_loss = 0.
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        for images in train_loader:
            images = images.cuda()
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            total_loss += loss.item()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder
            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
            )
        pbar.close()
        # save your improved network
        torch.save(resnet.state_dict(), './SSL_resnet50.pth')

def SL(args):
    fc_layer = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(True),
        nn.Linear(512, 65),
    )

    train_data = office_data(args.data_dir, args.label_path, 'train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

    val_data = office_data(args.data_dir, args.label_path, 'val')
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)

    resnet = models.resnet50(pretrained=False)
    if args.load:
        try:
            resnet.state_dict(torch.load(args.load))
            resnet.fc = fc_layer
        except:
            resnet.fc = fc_layer
            resnet.state_dict(torch.load(args.load))
    
    if args.fix_backbone:
        print('Fix backbones!')
        ct = 0
        for child in resnet.children():
            ct += 1
            if ct < 8:
                for param in child.parameters():
                    param.requires_grad = False
    
    resnet = resnet.cuda()

    if args.opt == 'adam':
        opt = torch.optim.Adam(resnet.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.opt == 'sgd':
        opt = torch.optim.SGD(resnet.parameters(), lr=args.lr, weight_decay=5e-4, momentum = 0.9)
    scheduler = StepLR(opt, step_size=50, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    max_acc = 0
    print('Start training!')
    for epoch in range(args.n_epochs):
        resnet.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        label_total = 0
        loss_total = 0.
        correct_total = 0
        for images, label in train_loader:
            images = images.cuda()
            label = label.cuda()
            pred = resnet(images)
            loss = criterion(pred, label)
            loss_total += loss.item()

            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = (correct_total / label_total) * 100

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_total:.4f}",
            Accuracy=f"{acc:.2f}%"
            )
        pbar.close()

        # Validation
        resnet.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="validation[%d/%d]"%(epoch, args.n_epochs), unit=" step")
            label_total = 0
            loss_total = 0.
            correct_total = 0
            for images, label in val_loader:
                images = images.cuda()
                label = label.cuda()
                pred = resnet(images)
                loss = criterion(pred, label)
                loss_total += loss.item()

                pred = pred.cpu().detach().numpy()
                pred_label = np.argmax(pred, axis=1)
                correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
                label_total += len(pred_label)
                correct_total += correct
                val_acc = (correct_total / label_total) * 100

                pbar.update()
                pbar.set_postfix(
                loss=f"{loss_total:.4f}",
                Accuracy=f"{val_acc:.2f}%"
                )
        pbar.close()
        scheduler.step()
        if val_acc > max_acc:
            print('Save model!!')
            max_acc = val_acc
            torch.save(resnet.state_dict(), '%s/model_epoch%d_acc%.2f.pth' % (args.save_model_path, epoch, val_acc))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch_size for training')
    parser.add_argument('--n_cpu', default=4, type=int, help='number of workers')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate of the adam')
    parser.add_argument('--load', type=str, default='./SSL_resnet50.pth', help="Model checkpoint path")
    parser.add_argument('--data_dir', default='../../hw4_data/office', type=str, help="Training images directory")
    parser.add_argument('--opt', default='adam', type=str, help="adam/sgd")
    parser.add_argument('--label_path', default='./label.csv', type=str, help="Training images directory")
    parser.add_argument('--fix_backbone', default=0, type=int, help="Whether to fix backbone")
    parser.add_argument('--save_model_path', default='./checkpoints/SL', type=str, help="Path to save model")
    args = parser.parse_args()
    os.makedirs(args.save_model_path, exist_ok=True)

    # SSL(args)
    SL(args)

    

    