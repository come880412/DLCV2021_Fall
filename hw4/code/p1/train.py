from collections import defaultdict
import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.nn import parameter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import csv
import random
import numpy as np
import tqdm

from utils import worker_init_fn, GeneratorSampler, CategoriesSampler, loss_metric, count_acc, compute_val_acc, euclidean_distance
from dataset import MiniDataset_test, MiniImageNet_train
from model import Convnet, parametric_func

import warnings
warnings.filterwarnings("ignore")

# fix random seeds for reproducibility
SEED = 412
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def train(args, model, train_loader, valid_loader):
    metric = parametric_func(args.N_way).cuda()
    optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':metric.parameters()}], lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # metric = loss_metric()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    
    
    max_acc = 0.
    for epoch in range(1, args.n_epochs + 1):
        total_loss = 0.
        model.train()
        metric.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        for i, batch in enumerate(train_loader):
            data, _ = [_.cuda() for _ in batch]
            p = args.N_shot * args.N_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.N_shot, args.N_way, -1).mean(dim=0)

            label = torch.arange(args.N_way).repeat(args.N_query)
            label = label.type(torch.cuda.LongTensor)

            query_features = model(data_query)
            # logits = metric.euclidean_metric(query_features, proto)
            # logits = metric.cosine_similarity_metric(query_features, proto)
            distance = euclidean_distance(query_features, proto).view(75, -1)
            logits = metric(distance)

            loss = F.cross_entropy(logits, label)
            total_loss += loss.item()
            
            acc = count_acc(logits, label) * 100

            pbar.update()
            pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            Accuracy=f"{acc:.2f}%"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.close()

        val_acc = predict(args, model, valid_loader, metric, epoch)
        # val_acc = valid(args, model, valid_loader, metric, epoch)
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), '%s/%s/model_epoch%d_valacc%.2f.pth' % (args.saved_model_path, args.saved_name, epoch, val_acc))
            torch.save(metric.state_dict(), '%s/%s/metric_epoch%d_valacc%.2f.pth' % (args.saved_model_path, args.saved_name, epoch, val_acc))
            print('Save model!!')
        lr_scheduler.step()

def valid(args, model, valid_loader, metric, epoch):
    model.eval()
    metric.eval()
    acc_list = []
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Validation[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        for i, batch in enumerate(valid_loader):
            data, _ = [_.cuda() for _ in batch]
            p = args.N_shot * args.N_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.N_shot, args.N_way, -1).mean(dim=0)

            label = torch.arange(args.N_way).repeat(args.N_query)
            label = label.type(torch.cuda.LongTensor)

            query_features = model(data_query)
            # logits = metric.euclidean_metric(query_features, proto)
            # logits = metric.cosine_similarity_metric(query_features, proto)
            distance = euclidean_distance(query_features, proto).view(75, -1)
            logits = metric(distance)
            loss = F.cross_entropy(logits, label)

            acc = count_acc(logits, label)
            acc_list.append(acc)

            pbar.update()
            pbar.set_postfix(
            loss=f"{loss:.4f}",
            Accuracy=f"{acc * 100:.2f}%"
            )
    episodic_acc = np.array(acc_list)
    mean = episodic_acc.mean()
    std = episodic_acc.std()

    val_acc = mean * 100
    error = 1.96 * std / (600)**(1/2) * 100
    pbar.set_postfix(
    loss=f"{loss:.4f}",
    val_acc = f"{val_acc:.2f}%",
    error = f"{error:.2f}%"
    )
    return val_acc

def predict(args, model, valid_loader, metric, epoch):
    model.eval()
    metric.eval()
    gt_csv = np.loadtxt(args.valcase_gt_csv, delimiter=',', dtype=np.str)
    total_loss = 0.
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Validation[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        for i, (data, target) in enumerate(valid_loader):
            support_input = data[:5 * args.N_shot,:,:,:].cuda()
            query_input   = data[5 * args.N_shot:,:,:,:].cuda()

            label_encoder = {target[i * args.N_shot] : i for i in range(5)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[5 * args.N_shot:]])
            
            proto = model(support_input)
            proto = proto.reshape(args.N_shot, 5, -1).mean(dim=0)

            query_features = model(query_input)
            
            # logits = metric.euclidean_metric(query_features, proto)
            # logits = metric.cosine_similarity_metric(query_features, proto)
            # logits = metric(torch.cat((query_features, proto.view(-1).repeat(75).view(75, 1280)), dim=1))
            distance = euclidean_distance(query_features, proto).view(75, -1)
            logits = metric(distance)
            loss = F.cross_entropy(logits, query_label)
            total_loss += loss.item()

            pbar.update()
            pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            )
            pred = torch.argmax(logits, dim=1).cpu().detach().numpy()
            pred = pred.astype(str)

            gt_csv[i+1,1:] = pred
    np.savetxt(args.output_csv, gt_csv, fmt='%s', delimiter=',')

    val_acc, error = compute_val_acc(args.valcase_gt_csv, args.output_csv)
    pbar.set_postfix(
    loss=f"{total_loss:.4f}",
    val_acc = f"{val_acc:.2f}%",
    error = f"{error:.2f}%"
    )
    pbar.close()
    return val_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of epochs for training')
    parser.add_argument('--training_batch', default=100, type=int, help='Number of batch_size for training')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate of the adam')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--train_csv', type=str, default='../../hw4_data/mini/train.csv', help="Training images csv file")
    parser.add_argument('--train_data_dir', default='../../hw4_data/mini/train', type=str, help="Training images directory")
    parser.add_argument('--val_csv', type=str, default='../../hw4_data/mini/val.csv', help="Validation images csv file")
    parser.add_argument('--val_data_dir', type=str, default='../../hw4_data/mini/val', help="Validation images directory")
    parser.add_argument('--valcase_csv', type=str, default='../../hw4_data/mini/val_testcase.csv', help="Validation case csv")
    parser.add_argument('--valcase_gt_csv', type=str, default='../../hw4_data/mini/val_testcase_gt.csv', help="Validation case csv")
    parser.add_argument('--output_csv', type=str, default='./predict.csv', help="Output filename")
    parser.add_argument('--saved_model_path', type=str, default='./checkpoints', help="Path to save model")
    parser.add_argument('--saved_name', type=str, default='5way-1shot_parametric', help="name of your saved_model name")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    os.makedirs('%s/%s' % (args.saved_model_path, args.saved_name), exist_ok=True)

    train_dataset = MiniImageNet_train(args.train_csv, args.train_data_dir)

    train_sampler = CategoriesSampler(train_dataset.label, args.training_batch,
                                      args.N_way, args.N_shot + args.N_query)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler,
                              num_workers=4, pin_memory=True)

    # val_dataset = MiniImageNet_train(args.val_csv, args.val_data_dir)

    # val_sampler = CategoriesSampler(val_dataset.label, 600,
    #                                   args.N_way, args.N_shot + args.N_query)
    # val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
    #                           num_workers=4, pin_memory=True)

    val_dataset = MiniDataset_test(args.val_csv, args.val_data_dir)

    val_loader = DataLoader(
        val_dataset, batch_size= 5 * (args.N_query + args.N_shot),
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.valcase_csv))

    model = Convnet().cuda()

    train(args, model, train_loader, val_loader)
