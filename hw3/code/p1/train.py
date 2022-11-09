from torch._C import import_ir_module
from timm import create_model
import argparse
import os
from dataset import p1_data
from torch.utils.data import DataLoader
from utils import FocalLoss, get_cosine_schedule_with_warmup
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR
import torch
import tqdm
import numpy as np

def train(opt, model, train_loader, val_loader):
    writer = SummaryWriter('runs/%s' % opt.name)
    cuda = True if torch.cuda.is_available() else False
    if opt.loss == 'focal':
        print('Loss function: focal loss')
        criterion = FocalLoss(gamma=2)
    else:
        print('Loss function: crossentropy loss')
        criterion = torch.nn.CrossEntropyLoss()
    if opt.optimizer == 'adam':
        print('Optimizer: adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay = 5e-4)
    if opt.optimizer == 'sgd':
        print('Optimizer: sgd')
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay = 5e-4, momentum=0.9)
    if cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    """lr_scheduler"""
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - opt.lr_decay_epoch) / float(opt.lr_decay_epoch + 1)
        return lr_l
    if opt.scheduler == 'linear':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=opt.lr_decay_epoch, gamma=0.412)
    elif opt.scheduler == 'warmup':
        total_steps = len(train_loader) * opt.n_epochs
        warmup_steps = total_steps * 0.06
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    """training"""
    print('Start training!')
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    max_acc = 0.
    train_update = 0
    for epoch in range(opt.initial_epoch, opt.n_epochs):
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")
        loss_total = 0
        acc = 0.
        correct_total = 0
        label_total = 0
        for image, label, _ in train_loader:
            if cuda:
                image = image.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, label)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = (correct_total / label_total) * 100
            loss_total += loss
            loss.backward()
            optimizer.step()
            if opt.scheduler == 'warmup':
                scheduler.step()

            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_total:.4f}",
            Accuracy=f"{acc:.2f}%"
            )
            writer.add_scalar('training loss', loss, train_update)
            writer.add_scalar('training accuracy', acc, train_update)
            train_update += 1
        pbar.close()
        val_acc = validation(opt, model, val_loader, writer, epoch)
        if max_acc <= val_acc:
            print('save model!!')
            max_acc = val_acc
            torch.save(model.state_dict(), './checkpoints/%s/model_epoch%d_acc%.2f.pth' % (opt.name, epoch, max_acc))
        if opt.scheduler == 'step' or opt.scheduler == 'linear':
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
    print('best ACC:%.2f' % (max_acc))

def validation(opt, model, val_loader, writer, epoch):
    model.eval()
    cuda = True if torch.cuda.is_available() else False
    if opt.loss == 'focal':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()
    label_total = 0
    correct_total = 0
    loss_total = 0.
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="val", unit=" step")
    for image, label, _ in val_loader:
        with torch.no_grad():
            if cuda:
                image = image.cuda()
                label = label.cuda()
            pred = model(image)
            loss = criterion(pred, label)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = (correct_total / label_total) * 100
            loss_total += loss
            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_total:.4f}",
            Accuracy=f"{acc:.2f}"
            )
    writer.add_scalar('validation loss', loss_total, epoch)
    writer.add_scalar('validation accuracy', acc, epoch)
    pbar.close()
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")
    parser.add_argument("--lr_decay_epoch", type=int, default=30, help="Start to decay epoch")
    parser.add_argument('--train_data_path', default='../../hw3_data/p1_data/train', help='path to train data')
    parser.add_argument('--val_data_path', default='../../hw3_data/p1_data/val', help='path to validation data')
    parser.add_argument('--num_classes', type=int, default=37, help='number of classes')
    parser.add_argument('--loss', default='crossentropy', help='loss function(crossentropy/focal)')
    parser.add_argument('--optimizer', default='sgd', help='adam/sgd')
    parser.add_argument('--scheduler', default='step', help='linear/step/warmup')
    parser.add_argument('--n_cpu', default=8, help='number of cpu workers')
    parser.add_argument('--saved_model', default='', help='path to model to continue training')
    parser.add_argument('--name', default='', help='name of saved model folder')
    parser.add_argument('--model_name', default='vit_large_patch16_224', help='name of model')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")
    parser.add_argument("--gpu_id", type=str, default='0,1', help="gpu_id")
    opt = parser.parse_args()
    os.makedirs('./checkpoints/%s' % (opt.name), exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    train_data = p1_data(opt.train_data_path, 'train')
    train_loader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu)
    val_data = p1_data(opt.val_data_path, 'val')
    val_loader = DataLoader(val_data, batch_size=opt.batch_size,shuffle=False, num_workers=opt.n_cpu)

    model = create_model(opt.model_name, pretrained=True)
    model.head = nn.Linear(in_features= model.head.in_features, out_features=opt.num_classes, bias=True)
    model = nn.DataParallel(model)
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        model.load_state_dict(torch.load(opt.saved_model))
    train(opt, model, train_loader, val_loader)