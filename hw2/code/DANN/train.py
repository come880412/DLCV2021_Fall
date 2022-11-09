import torch
from torch.autograd.grad_mode import enable_grad
import torch.nn as nn
import numpy as np
import argparse

from dataset import DANN_data
from model import DANN
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import os
from torch.optim import Adam, SGD
from utils import FocalLoss, optimizer_scheduler

use_amp = True

def test(args, source_loader_val, target_loader_val, model, writer, epoch, dann):
    model.eval()
    if args.loss == 'crossentropy':
        criterion_class = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'focal':
        criterion_class = FocalLoss(gamma=2).cuda()

    label_source_total = 0
    correct_source_total = 0
    label_target_total = 0
    correct_target_total = 0

    loss_source_total = 0.
    loss_target_total = 0.
    pbar = tqdm.tqdm(total=len(source_loader_val), ncols=0, desc="Source", unit=" step")
    with torch.no_grad():
        for source_data in source_loader_val:
            source_image, source_label = source_data
            source_image, source_label = torch.FloatTensor(source_image).cuda(), torch.LongTensor(source_label).cuda()

            class_out = model(source_image)
            loss = criterion_class(class_out, source_label)
            loss_source_total += loss
            pred = class_out.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(source_label.cpu().numpy(), pred_label))
            label_source_total += len(pred_label)
            correct_source_total += correct
            source_acc = (correct_source_total / label_source_total) * 100
            pbar.update()
            pbar.set_postfix(
            Source_val_loss=f"{loss_source_total:.4f}",
            Source_val_acc=f"{source_acc:.2f}%"
            )
        pbar.close()

        if dann:
            label_domain_total = 0
            correct_domain_total = 0
            total_loss_domain = 0.
            criterion_domain = nn.CrossEntropyLoss().cuda()
            if len(source_loader_val) > len(target_loader_val):
                val_len = len(target_loader_val)
            else:
                val_len = len(source_loader_val)
            pbar.close()
            pbar = tqdm.tqdm(total=val_len, ncols=0, desc="Domain", unit=" step")
            for batch_idx, (source_data, target_data) in enumerate(zip(source_loader_val, target_loader_val)):
                p = float(batch_idx) / val_len
                alpha = 2. / (1. + np.exp(-10 * p)) - 1 

                source_image, source_label = source_data
                target_image, _ = target_data
                source_image, source_label = torch.FloatTensor(source_image).cuda(), torch.LongTensor(source_label).cuda()
                target_image = torch.FloatTensor(target_image).cuda()
                domain_source_labels = torch.zeros(source_image.shape[0]).type(torch.LongTensor).cuda()
                domain_target_labels = torch.ones(target_image.shape[0]).type(torch.LongTensor).cuda()
                

                # Compute loss
                _, domain_source_out = model(source_image, True, alpha)
                loss_domain_soruce = criterion_domain(domain_source_out, domain_source_labels)

                _, domain_target_out = model(target_image, True, alpha)
                loss_domain_target = criterion_domain(domain_target_out, domain_target_labels)

                loss_domain = loss_domain_soruce + loss_domain_target
                total_loss_domain += loss_domain

                domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0)
                domain_out = torch.cat((domain_source_out, domain_target_out), dim=0)
                pred_domain = domain_out.cpu().detach().numpy()
                pred_domain_label = np.argmax(pred_domain, axis=1)
                correct = np.sum(np.equal(domain_combined_label.cpu().numpy(), pred_domain_label))
                label_domain_total += len(pred_domain_label)
                correct_domain_total += correct
                domain_acc = (correct_domain_total / label_domain_total) * 100

                pbar.update()
                pbar.set_postfix(
                Domain_val_loss=f"{total_loss_domain:.4f}",
                Domain_val_acc=f"{domain_acc:.2f}%",
                )
    pbar.close()

    if dann:
        writer.add_scalar('Domain val loss', total_loss_domain, epoch)
        writer.add_scalar('Domain val acc', domain_acc, epoch)
    writer.add_scalar('Source val loss', loss_source_total, epoch)
    writer.add_scalar('Source val acc', source_acc, epoch)
    writer.add_scalar('Target val loss', loss_target_total, epoch)
    
    if dann:
        return source_acc, domain_acc
    else:
        return source_acc

def train_source_target_only(args, model, source_loader_train, source_loader_val, target_loader_train, target_loader_val):
    writer = SummaryWriter('runs/%s/%s' % (args.transfer_mode, args.source))
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr = args.lr)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr = args.lr, momentum=0.9)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.loss == 'crossentropy':
        criterion_class = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'focal':
        criterion_class = FocalLoss(gamma=2).cuda()

    def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.lr_decay) / float(args.lr_decay + 1)
            return lr_l
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
    
    best_acc = 0
    train_update = 0
    for epoch in range(args.n_epochs):
        pbar = tqdm.tqdm(total=len(source_loader_train), ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        model.train()
        label_total = 0
        correct_total = 0
        loss_total = 0.

        start_steps = epoch * len(source_loader_train)
        total_steps = args.n_epochs * len(source_loader_train)

        for batch_idx, source_data in enumerate(source_loader_train):
            p = float(batch_idx + start_steps) / total_steps
            source_image, source_label = source_data
            source_image, source_label = torch.FloatTensor(source_image), torch.LongTensor(source_label)
            source_image, source_label = source_image.cuda(), source_label.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                class_out = model(source_image)
                loss = criterion_class(class_out, source_label)
            scaler.scale(loss).backward()
            loss_total += loss
            scaler.step(optimizer)
            scaler.update()
            # Calculate source acc
            pred = class_out.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(source_label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            soruce_acc = (correct_total / label_total) * 100

            pbar.update()
            pbar.set_postfix(
            Source_train_loss=f"{loss_total:.4f}",
            Source_train_acc=f"{soruce_acc:.2f}%"
            )
            writer.add_scalar('Source train loss', loss, train_update)
            writer.add_scalar('Source train acc', soruce_acc, train_update)
            train_update += 1
        pbar.close()
        source_val_acc = test(args, source_loader_val, target_loader_val, model, writer, epoch, False)
        if best_acc < source_val_acc:
            best_acc = source_val_acc
            print('Save model!!')
            torch.save(model.state_dict(), './checkpoints/%s/%s/model_epoch%d_%s_%s_soruceacc%.2f.pth' % 
            (args.transfer_mode, args.source, epoch, args.source, args.target, source_val_acc))
        scheduler.step()

def train_transfer(args, model, source_loader_train, source_loader_val, target_loader_train, target_loader_val):
    writer = SummaryWriter('runs/%s/%s' % (args.transfer_mode, args.source))
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr = args.lr)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr = args.lr, momentum=0.9)
    else:
        raise("Please specify an optimizer")

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.loss == 'crossentropy':
        criterion_class = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'focal':
        criterion_class = FocalLoss(gamma=2).cuda()
    criterion_domain = nn.CrossEntropyLoss().cuda()
    
    def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.lr_decay) / float(args.lr_decay + 1)
            return lr_l
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)

    best_acc = 0
    train_update = 0
    count_source = 0

    source_len = len(source_loader_train)
    target_len = len(target_loader_train)
    target_iter = iter(target_loader_train)

    for epoch in range(args.n_epochs):
        pbar = tqdm.tqdm(total=source_len, ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        model.train()

        label_source_total = 0
        correct_source_total = 0
        label_domain_total = 0
        correct_domain_total = 0

        start_steps = epoch * source_len
        total_steps = args.n_epochs * source_len
        
        for batch_idx, source_data in enumerate(source_loader_train):
            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            if target_len == count_source:
                count_source =0 
                target_iter = iter(target_loader_train)
            target_data = target_iter.next()
            count_source += 1

            source_image, source_label = source_data
            target_image, _ = target_data
            source_image, source_label = torch.FloatTensor(source_image).cuda(), torch.LongTensor(source_label).cuda()
            target_image = torch.FloatTensor(target_image).cuda()
            domain_source_labels = torch.zeros(source_image.shape[0]).type(torch.LongTensor).cuda()
            domain_target_labels = torch.ones(target_image.shape[0]).type(torch.LongTensor).cuda()

            optimizer.zero_grad()

            # Compute loss
            with torch.cuda.amp.autocast(enabled=use_amp):
                class_out, domain_source_out = model(source_image, True, alpha)
                loss_classification = criterion_class(class_out, source_label)
                loss_domain_soruce = criterion_domain(domain_source_out, domain_source_labels)

                _, domain_target_out = model(target_image, True, alpha)
                loss_domain_target = criterion_domain(domain_target_out, domain_target_labels)

            loss_domain = loss_domain_soruce + loss_domain_target

            loss = loss_classification + loss_domain
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            domain_out = torch.cat((domain_source_out, domain_target_out), dim=0)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), dim=0)
            # Metrices calculation
            pred = class_out.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(source_label.cpu().numpy(), pred_label))
            label_source_total += len(pred_label)
            correct_source_total += correct
            soruce_acc = (correct_source_total / label_source_total) * 100

            pred_domain = domain_out.cpu().detach().numpy()
            pred_domain_label = np.argmax(pred_domain, axis=1)
            correct = np.sum(np.equal(domain_combined_label.cpu().numpy(), pred_domain_label))
            label_domain_total += len(pred_domain_label)
            correct_domain_total += correct
            domain_acc = (correct_domain_total / label_domain_total) * 100

            pbar.update()
            pbar.set_postfix(
            Source_loss=f"{loss_classification:.4f}",
            Source_acc=f"{soruce_acc:.2f}%",
            Domain_loss=f"{loss_domain:.4f}",
            Domain_acc=f"{domain_acc:.2f}%",
            )
            writer.add_scalar('Source loss', loss_classification, train_update)
            writer.add_scalar('Source acc', soruce_acc, train_update)
            writer.add_scalar('Domain loss', loss_domain, train_update)
            writer.add_scalar('Domain acc', domain_acc, train_update)
            train_update += 1
        pbar.close()
        source_val_acc, domain_val_acc = test(args, source_loader_val, target_loader_val, model, writer, epoch, True)
        torch.save(model.state_dict(), './checkpoints/%s/%s/model_epoch%d_%s_%s_sourceacc%.2f_domainacc%.2f.pth'%
        (args.transfer_mode, args.source, epoch, args.source, args.target, source_val_acc, domain_val_acc))
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN implement')
    parser.add_argument('--data_root', type=str, default='../../hw2_data/digits')
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd/adam')
    parser.add_argument('--transfer_mode', type=str, default='transfer', help='source_only/target_only/transfer')
    parser.add_argument('--saved_model', type=str, default='./checkpoints/transfer/usps/model_epoch47_usps_svhn_sourceacc98.42_domainacc69.55.pth', help='path to saved_model to continue training')
    parser.add_argument('--loss', default='crossentropy', help='loss function(crossentropy/focal)')
    parser.add_argument('--source', type=str, default= 'usps') # svhn, mnistm, usps
    parser.add_argument('--target', type=str, default='svhn') #mnistm, usps, svhn
    parser.add_argument('--gpu_id', type=str, default= '0')
    parser.add_argument('--lr', type=float, default= 1e-4)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr_decay', type=int, default=50)
    parser.add_argument('--n_cpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs('./checkpoints/%s/%s' % (args.transfer_mode, args.source), exist_ok=True)
    
    args.source_csv = './data_split/%s' % args.source
    args.target_csv = './data_split/%s' % args.target
    source_data_root = os.path.join(args.data_root, args.source)
    target_data_root = os.path.join(args.data_root, args.target)

    # Data loader
    data_source_train = DANN_data(source_data_root, args.source_csv, 'train')
    data_source_val = DANN_data(source_data_root, args.source_csv, 'val')
    data_target_train = DANN_data(target_data_root, args.target_csv, 'train')
    data_target_val = DANN_data(target_data_root, args.target_csv, 'val')

    source_loader_train = DataLoader(data_source_train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus)
    source_loader_test = DataLoader(data_source_val, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus)
    target_loader_train = DataLoader(data_target_train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus)
    target_loader_test = DataLoader(data_target_val, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus)

    model = DANN().cuda()
    if args.saved_model:
        model.load_state_dict(torch.load(args.saved_model))
        print('Load pre-trained model')

    if args.transfer_mode == 'source_only' or args.transfer_mode == 'target_only':
        train_source_target_only(args, model, source_loader_train, source_loader_test, target_loader_train, target_loader_test)
    else:
        train_transfer(args, model, source_loader_train, source_loader_test, target_loader_train, target_loader_test)