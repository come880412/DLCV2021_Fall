import torch
import argparse
import logging
from dataset import DANN_data
from model_TSNE import DANN
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

color_list = ['peru', 'dodgerblue', 'brown', 'darkslategray', 'lightsalmon', 'orange', 'aquamarine', 'springgreen', 'chartreuse', 'fuchsia',
	      'mediumspringgreen', 'burlywood', 'palegreen', 'orangered', 'lightcoral', 'tomato', 'pink', 'darkseagreen', 'olive', 'darkgoldenrod',
              'turquoise', 'plum', 'darkmagenta', 'deeppink', 'red', 'slategrey', 'darkviolet', 'darkturquoise', 'skyblue', 'mediumorchid',
	      'magenta', 'deepskyblue', 'darkorchid', 'teal', 'wheat', 'green', 'lightcyan', 'royalblue', 'sienna', 'seagreen', 
	      'blueviolet', 'darkorange', 'aqua', 'purple', 'darkred', 'salmon', 'orchid', 'lightgreen', 'cadetblue', 'thistle']

def feature_extract_t(args, model):
    model.eval()
    feature_list = []
    label_list = []
    data_test = DANN_data('../../hw2_data/digits/%s'%args.target, mode='test')
    dataloader = DataLoader(data_test, batch_size=128, shuffle=False, num_workers=2)
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            images, labels = data
            images = images.cuda()
            label_feature = model(images)
            label_feature = label_feature.cpu().detach().numpy()
            for i in range(len(label_feature)):
                feature_list.append(label_feature[i])
                label_list.append(labels[i])
    return np.array(feature_list), np.array(label_list)

def feature_extract_s_t(args, model):
    model.eval()
    feature_list = []
    label_list = []
    data_source = DANN_data('../../hw2_data/digits/%s'%args.source, mode='test')
    dataloader_s = DataLoader(data_source, batch_size=128, shuffle=False, num_workers=2)
    data_target = DANN_data('../../hw2_data/digits/%s'%args.target, mode='test')
    dataloader_t = DataLoader(data_target, batch_size=128, shuffle=False, num_workers=2)
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader_s):
            images, _ = data
            images = images.cuda()
            domain_feature = model(images)
            domain_feature = domain_feature.cpu().detach().numpy()
            for i in range(len(domain_feature)):
                feature_list.append(domain_feature[i])
                label_list.append(0)
                
        
        for data in tqdm.tqdm(dataloader_t):
            images, _ = data
            images = images.cuda()
            _, domain_feature = model(images)
            domain_feature = domain_feature.cpu().detach().numpy()
            for i in range(len(domain_feature)):
                feature_list.append(domain_feature[i])
                label_list.append(1)
    return np.array(feature_list), np.array(label_list)


def classifer_tsne(args, feature, label):
    x = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    y = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    ratio_print = 0.05
    tsne = TSNE(n_components=2, init='random', random_state=412, verbose=1, n_iter=1000, n_jobs=4)
    X_tsne = tsne.fit_transform(feature)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        x[label[i]].append(X_norm[i, 0])
        y[label[i]].append(X_norm[i, 1])

    for i in range(len(x)):
        plt.scatter(x[i][:int(len(x[i])*ratio_print)], y[i][:int(len(x[i])*ratio_print)], color=color_list[i], s=3)
        plt.scatter(x[i][:], y[i][:], color=color_list[i], s=3)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./t_sne_result/%s_%s_classfier.png' % (args.source, args.target))
    # plt.show()

def domain_tsne(args, feature, label):
    x = {0:[], 1:[]}
    y = {0:[], 1:[]}
    ratio_print = 0.2

    tsne = TSNE(n_components=2, init='random', random_state=412, verbose=1, n_iter=10000, n_jobs=5)
    X_tsne = tsne.fit_transform(feature)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        x[label[i]].append(X_norm[i, 0])
        y[label[i]].append(X_norm[i, 1])
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]), 
        #         fontdict={'weight': 'bold', 'size': 9})
    for i in range(len(x)):
        plt.scatter(x[i][:int(len(x[i])*ratio_print)], y[i][:int(len(x[i])*ratio_print)], color=plt.cm.Set1(i), s=3)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./t_sne_result/%s_%s_domain.png' % (args.source, args.target))
    # plt.show()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='DANN implement')
    parser.add_argument('--model_path',type=str,default='./checkpoints/transfer/usps/model_epoch10_usps_svhn_sourceacc96.46_targetacc29.04_domainacc77.36.pth')
    parser.add_argument('--source',type=str,default= 'usps') # mnistm, svhn, usps
    parser.add_argument('--target',type=str,default='svhn') # usps, mnistm, svhn
    parser.add_argument('--GPUID',type=str,default= '0')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPUID
    os.makedirs('./t_sne_result', exist_ok=True)
    model = DANN().cuda()
    model.load_state_dict(torch.load(args.model_path))
    
    feature_cls, label_cls = feature_extract_t(args, model)
    feature_domain, label_domain = feature_extract_s_t(args, model)
    print(feature_cls.shape)
    classifer_tsne(args, feature_cls, label_cls)
    # domain_tsne(args, feature_domain, label_domain)
    
