from random import sample
from numpy.core.fromnumeric import size
import torch
from torch.utils.data import DataLoader, sampler
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import os
import argparse
import tqdm
from torchsampler import ImbalancedDatasetSampler
import random

from dataset import food_data
from arcmodel.model import resnet50, ArcSEresnet50, ArcSEresnet101, ArcEfficientNet, Arcresnest50, Arcresnest101
from utils import ArcMarginProduct, Balanced_CE, label_to_freq

def train(opt, model, metric_fc, criterion, criterionClass, optimizer, val_loader, label_freq_dict):
    writer = SummaryWriter('runs/%s' % opt.saved_name)

    max_acc = float('-inf')
    train_update = 0
    for epoch in range(opt.initial_epoch, opt.n_epochs):
        model.train()
        metric_fc.train()

        if opt.semi:
            train_data = food_data(opt.data_path, 'semi', opt.image_size)
            train_loader = DataLoader(train_data, 
                                batch_size=opt.batch_size, 
                                num_workers=opt.n_cpu, 
                                pin_memory=opt.pin_memory, drop_last=opt.drop_last, sampler=ImbalancedDatasetSampler(train_data, opt.reweight))
        else:
            train_data = food_data(opt.data_path, 'train', opt.image_size)
            train_loader = DataLoader(train_data, 
                                    batch_size=opt.batch_size, 
                                    num_workers=opt.n_cpu, 
                                    pin_memory=opt.pin_memory, drop_last=opt.drop_last, sampler=ImbalancedDatasetSampler(train_data, opt.reweight))

        print('# of training data: ', len(train_data))
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader))

        total_loss = 0.
        total_freq = {'all':0, 'r':0, 'c':0, 'f':0}
        correct_freq = {'all':0, 'r':0, 'c':0, 'f':0}
        for image, label, _ in train_loader:
            image, label = image.cuda(), label.cuda()
            optimizer.zero_grad()
            
            if opt.arc:
                pred, embedding = model(image)
                embedding = metric_fc(embedding, label)
                loss1 = criterion(embedding, label)
                loss2 = criterionClass(pred, label)
                loss = loss1 + loss2
                loss.backward()
            else:
                pred, _ = model(image)
                loss = criterionClass(pred, label)
                loss.backward()

            optimizer.step()

            total_loss += loss.item()
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            total_freq, correct_freq = label_to_freq(pred_label, label.cpu().numpy(), label_freq_dict, total_freq, correct_freq)

            acc_all = (correct_freq['all'] / total_freq['all']) * 100
            acc_f = (correct_freq['f'] / (total_freq['f'] + 1e-6)) * 100
            acc_c = (correct_freq['c'] / (total_freq['c'] + 1e-6)) * 100
            acc_r = (correct_freq['r'] / (total_freq['r'] + 1e-6)) * 100


            pbar.update()
            pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            acc_all=f"{acc_all:.2f}%",
            acc_f=f"{acc_f:.2f}%",
            acc_c=f"{acc_c:.2f}%",
            acc_r=f"{acc_r:.2f}%",
            lr = f"{optimizer.param_groups[0]['lr']:.6f}"
            )
            writer.add_scalar('Training loss', loss, train_update)
            writer.add_scalar('Training acc_all', acc_all, train_update)
            writer.add_scalar('Training acc_f', acc_f, train_update)
            writer.add_scalar('Training acc_c', acc_c, train_update)
            writer.add_scalar('Training acc_r', acc_r, train_update)
            train_update += 1
            scheduler.step()
        pbar.close()
        if opt.val:
            val_acc_all, val_acc_f, val_acc_c, val_acc_r = validation(opt, model, metric_fc, criterion, val_loader, epoch, writer, label_freq_dict)
            if val_acc_all > max_acc:
                max_acc = val_acc_all
                torch.save(model.state_dict(), '%s/%s/model_epoch%d_all%.2f_f%.2f_c%.2f_r%.2f.pth' % (opt.saved_model, opt.saved_name, epoch, val_acc_all, val_acc_f, val_acc_c, val_acc_r))
                if opt.arc:
                    torch.save(metric_fc.state_dict(), '%s/%s/metric_epoch%d_all%.2f_f%.2f_c%.2f_r%.2f.pth' % (opt.saved_model, opt.saved_name, epoch, val_acc_all, val_acc_f, val_acc_c, val_acc_r))
                print('Save model!!')
        else:
            if acc_all > max_acc:
                max_acc = acc_all
                torch.save(model.state_dict(), '%s/%s/model_best.pth' % (opt.saved_model, opt.saved_name))
                if opt.arc:
                    torch.save(metric_fc.state_dict(), '%s/%s/metric_best.pth' % (opt.saved_model, opt.saved_name))
                print('Save model!!')

        torch.save(model.state_dict(), '%s/%s/model_last.pth' % (opt.saved_model, opt.saved_name))
        if opt.arc:
            torch.save(metric_fc.state_dict(), '%s/%s/metric_last.pth' % (opt.saved_model, opt.saved_name))
        
def validation(opt, model, metric_fc, criterion, val_loader, epoch, writer, label_freq_dict):
    model.eval()
    metric_fc.eval()
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="Validation[%d/%d]"%(epoch, opt.n_epochs), unit=" step")

    val_loss = 0.
    total_freq = {'all':0, 'r':0, 'c':0, 'f':0}
    correct_freq = {'all':0, 'r':0, 'c':0, 'f':0}
    with torch.no_grad():
        for image, label, _ in val_loader:
            image, label = image.cuda(), label.cuda()

            if opt.arc:
                pred, embedding = model(image)
                embedding = metric_fc(embedding, label)
                loss1 = criterion(embedding, label)
                loss2 = criterionClass(pred, label)
                loss = loss1 + loss2
            else:
                pred, _ = model(image)
                loss = criterionClass(pred, label)

            val_loss += loss.item()
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            total_freq, correct_freq = label_to_freq(pred_label, label.cpu().numpy(), label_freq_dict, total_freq, correct_freq)

            acc_all = (correct_freq['all'] / total_freq['all']) * 100
            acc_f = (correct_freq['f'] / (total_freq['f'] + 1e-6)) * 100
            acc_c = (correct_freq['c'] / (total_freq['c'] + 1e-6)) * 100
            acc_r = (correct_freq['r'] / (total_freq['r'] + 1e-6)) * 100

            pbar.update()
            pbar.set_postfix(
                loss=f"{val_loss:.4f}",
                acc_all=f"{acc_all:.2f}%",
                acc_f=f"{acc_f:.2f}%",
                acc_c=f"{acc_c:.2f}%",
                acc_r=f"{acc_r:.2f}%",

                )
    writer.add_scalar('Validation loss', val_loss, epoch)
    writer.add_scalar('Validation acc_all', acc_all, epoch)
    writer.add_scalar('Validation acc_f', acc_f, epoch)
    writer.add_scalar('Validation acc_c', acc_c, epoch)
    writer.add_scalar('Validation acc_r', acc_r, epoch)
    pbar.close()
    return acc_all, acc_f, acc_c, acc_r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument('--image_size', type=int, default = 384, help='The size of image')
    parser.add_argument('--model', default='Arcresnest101', help='ArcSEResnet50/ArcSEResnet101/EfficientNet/Arcresnest50/Arcresnest101')

    parser.add_argument('--EfficientNet_mode', type=str, default="efficientnet-b4", help='Whether to use ImageNet pretrained weight')
    parser.add_argument('--feature', type=int, default = 1792, help='B0:1280, B3:1536, B4:1792, B5:2048, B6:2304')

    parser.add_argument('--data_path', default='../dataset', help='path to data')
    parser.add_argument('--saved_name', default='arc_margin', help='name of saved model folder name')
    parser.add_argument('--saved_model', default='./checkpoints', help='path to save model')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu workers')
    parser.add_argument('--pin_memory', type=bool, default=True, help='If True, the data loader will copy Tensors into CUDA pinned memory before returning them.')
    parser.add_argument('--drop_last', type=bool, default=True, help='set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.')

    parser.add_argument('--model_load', default='', help='path to model to continue training')
    parser.add_argument('--metric_load', default='', help='path to metric to continue training')

    parser.add_argument('--adam', action = 'store_true', help='use torch.optim.Adam() or torch.optim.SGD() optimizer')
    parser.add_argument('--momentum', type=float, default = 0.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default = 5e-4, help='optimizer weight_decay')
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument('--val', action = 'store_true', help='whether to use validation set')

    parser.add_argument('--arc', action = 'store_true', help="whether to use arc_margin")
    parser.add_argument('--ArcFeature', type=int, default = 512, help='How many features are input to ArcMarginProduct')
    parser.add_argument('--num_classes', type=int, default = 1000, help='How many categories to predict')
    parser.add_argument('--ArcS', type=float, default = 30, help="feature scale s")
    parser.add_argument('--ArcM', type=float, default = 0.5, help="angular margin m")
    parser.add_argument('--easy_margin', type=bool, default = False)

    parser.add_argument('--reweight', action = 'store_true', help='whether to reweight the sampler')
    parser.add_argument('--semi', action = 'store_true', help='use validation or not')
    parser.add_argument('--gamma', type=float, default = 2, help='use Focal loss gamma')
    parser.add_argument('--gammaClass', type=float, default = 1.5, help='use Focal loss gamma(class loss)')
    opt = parser.parse_args()
    os.makedirs('%s/%s' % (opt.saved_model, opt.saved_name), exist_ok=True)

    label_freq_dict = {}
    label_name = np.loadtxt('%s/label2name.txt' % (opt.data_path), dtype=str, encoding="utf-8")
    for data in label_name:
        label = data[0]
        freq = data[1]
        label_freq_dict[int(label)] = freq

    val_data = food_data(opt.data_path, 'val', opt.image_size)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=opt.n_cpu, pin_memory=opt.pin_memory)

    # print('# of training data: ', len(train_data))
    print('# of validation data: ', len(val_data))
    
    if opt.model == 'ArcSEResnet50':
        model = ArcSEresnet50(num_classes = opt.num_classes, ArcFeature = opt.ArcFeature, pretrained = True)
    elif opt.model == 'ArcSEResnet101':
        model = ArcSEresnet101(num_classes = opt.num_classes, ArcFeature = opt.ArcFeature, pretrained = True)
    elif opt.model == "EfficientNet":
        model = ArcEfficientNet(mode = opt.EfficientNet_mode, \
                                advprop = False, \
                                num_classes = opt.num_classes, \
                                feature = opt.feature, \
                                ArcFeature = opt.ArcFeature
                                )
    elif opt.model == 'Arcresnest50':
        model = Arcresnest50(opt.num_classes, opt.ArcFeature, True)
    elif opt.model == 'Arcresnest101':
        model = Arcresnest101(opt.num_classes, opt.ArcFeature, True)

    model = model.cuda()

    metric_fc = ArcMarginProduct(opt.ArcFeature, opt.num_classes, s = opt.ArcS, m = opt.ArcM, easy_margin = opt.easy_margin)
    

    if opt.model_load:
        print('load pretrained model!!')
        model.load_state_dict(torch.load(opt.model_load))
        if opt.arc:
            metric_fc.load_state_dict(torch.load(opt.metric_load))

    criterion = Balanced_CE(gamma = opt.gamma) #2.0
    criterionClass = Balanced_CE(gamma = opt.gammaClass) #1.5

    if opt.arc:
        metric_fc = metric_fc.cuda()
        if opt.adam:
            print('Optimizer: Adam!!!')
            optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                        lr = opt.lr, weight_decay = opt.weight_decay)
        else:
            print('Optimizer: SGD!!!')
            optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                        lr = opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)
    else:
        if opt.adam:
            print('Optimizer: Adam!!!')
            optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr, weight_decay = opt.weight_decay)
        else:
            print('Optimizer: SGD!!!')
            optimizer = torch.optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)

    print('Start training!')
    train(opt, model, metric_fc, criterion, criterionClass ,optimizer, val_loader, label_freq_dict)


    