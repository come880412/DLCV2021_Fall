from re import purge
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


import numpy as np
import os
import argparse
import tqdm


from dataset import food_data
from model.model import resnet50, resnet101, ensemble_net, resnest50, resnest101, SEresnet50, SEresnet101, Efficinet_net
from model.pvtv2.model import pvt_v2_b2, pvt_v2_b3
from arcmodel.model import ArcSEresnet50

from utils import label_to_freq, FocalLoss

import warnings
warnings.filterwarnings("ignore")

def test_aug_ensemble(opt, model, test_loader):
    model.eval()
    csv_all = np.loadtxt(os.path.join(opt.csv_path, 'sample_submission_main_track.csv'), delimiter=',', dtype=np.str)
    csv_freq = np.loadtxt(os.path.join(opt.csv_path, 'sample_submission_freq_track.csv'), delimiter=',', dtype=np.str)
    csv_comm = np.loadtxt(os.path.join(opt.csv_path, 'sample_submission_comm_track.csv'), delimiter=',', dtype=np.str)
    csv_rare = np.loadtxt(os.path.join(opt.csv_path, 'sample_submission_rare_track.csv'), delimiter=',', dtype=np.str)

    count_all = 1
    count_freq = 1
    count_comm = 1
    count_rare = 1
    with torch.no_grad():
        for image, image_name in tqdm.tqdm(test_loader):
            image = image.cuda()
            for i, image_tensor in enumerate(image):
                if opt.model_ensemble:
                    pred = model(image_tensor)
                else:
                    pred = model(image_tensor)
                    pred = torch.mean(pred, dim=0)
                label = torch.argmax(pred)
                label = label.cpu().detach().numpy()

                name = image_name[i]
                csv_all[count_all][1] = str(label)
                
                if count_freq < len(csv_freq):
                    if name == csv_freq[count_freq][0]:
                        csv_freq[count_freq][1] = str(label)
                        count_freq += 1

                if count_comm < len(csv_comm):
                    if name == csv_comm[count_comm][0]:
                        csv_comm[count_comm][1] = str(label)
                        count_comm += 1
                if count_rare < len(csv_rare):
                    if name == csv_rare[count_rare][0]:
                        csv_rare[count_rare][1] = str(label)
                        count_rare += 1
                count_all += 1
    np.savetxt(os.path.join(opt.output_path, 'pred_main_track.csv'), csv_all, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(opt.output_path, 'pred_comm_track.csv'), csv_comm, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(opt.output_path, 'pred_freq_track.csv'), csv_freq, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(opt.output_path, 'pred_rare_track.csv'), csv_rare, fmt='%s', delimiter=',')

def test(opt, model, test_loader):
    model.eval()
    csv_all = np.loadtxt(os.path.join(opt.csv_path, 'sample_submission_main_track.csv'), delimiter=',', dtype=np.str)
    csv_freq = np.loadtxt(os.path.join(opt.csv_path, 'sample_submission_freq_track.csv'), delimiter=',', dtype=np.str)
    csv_comm = np.loadtxt(os.path.join(opt.csv_path, 'sample_submission_comm_track.csv'), delimiter=',', dtype=np.str)
    csv_rare = np.loadtxt(os.path.join(opt.csv_path, 'sample_submission_rare_track.csv'), delimiter=',', dtype=np.str)

    count_all = 1
    count_freq = 1
    count_comm = 1
    count_rare = 1
    with torch.no_grad():
        for image, image_name in tqdm.tqdm(test_loader):
            image = image.cuda()
            pred = model(image)

            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            for i, label in enumerate(pred_label):
                name = image_name[i]
                csv_all[count_all][1] = str(label)
                
                if count_freq < len(csv_freq):
                    if name == csv_freq[count_freq][0]:
                        csv_freq[count_freq][1] = str(label)
                        count_freq += 1

                if count_comm < len(csv_comm):
                    if name == csv_comm[count_comm][0]:
                        csv_comm[count_comm][1] = str(label)
                        count_comm += 1
                if count_rare < len(csv_rare):
                    if name == csv_rare[count_rare][0]:
                        csv_rare[count_rare][1] = str(label)
                        count_rare += 1
                count_all += 1
    np.savetxt(os.path.join(opt.output_path, 'pred_main_track.csv'), csv_all, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(opt.output_path, 'pred_comm_track.csv'), csv_comm, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(opt.output_path, 'pred_freq_track.csv'), csv_freq, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(opt.output_path, 'pred_rare_track.csv'), csv_rare, fmt='%s', delimiter=',')

def val(opt, model, val_loader):
    criterion = FocalLoss(num_classes=1000, gamma = 2.0)

    label_freq_dict = {}
    label_name = np.loadtxt('%s/label2name.txt' % (opt.data_path), dtype=str, encoding="utf-8")
    for data in label_name:
        label = data[0]
        freq = data[1]
        label_freq_dict[int(label)] = freq

    model.eval()
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="Validation", unit=" step")

    val_loss = 0.
    total_freq = {'all':0, 'r':0, 'c':0, 'f':0}
    correct_freq = {'all':0, 'r':0, 'c':0, 'f':0}
    with torch.no_grad():
        for image, label, _ in val_loader:
            image, label = image.cuda(), label.cuda()

            pred = model(image)
            loss = criterion(pred, label)

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
    pbar.close()
    return acc_all, acc_f, acc_c, acc_r

def semi_supervised(opt, model, val_loader):
    criterion = FocalLoss(num_classes=1000, gamma = 2.0)

    semi_data = []
    label_freq_dict = {}
    label_name = np.loadtxt('%s/label2name.txt' % (opt.data_path), dtype=str, encoding="utf-8")
    for data in label_name:
        label = data[0]
        freq = data[1]
        label_freq_dict[int(label)] = freq

    model.eval()
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="Validation", unit=" step")

    val_loss = 0.
    total_freq = {'all':0, 'r':0, 'c':0, 'f':0}
    correct_freq = {'all':0, 'r':0, 'c':0, 'f':0}
    with torch.no_grad():
        for image, label, image_name in val_loader:
            image, label = image.cuda(), label.cuda()

            pred = model(image)
            loss = criterion(pred, label)

            val_loss += loss.item()

            pred = pred.cpu().detach().numpy()
            label = label.cpu().numpy()
            pred_label = np.argmax(pred, axis=1)

            for i in range(len(pred_label)):
                name = image_name[i]
                pseudo_label = pred_label[i]
                truth_label = label[i]
                prob = round(pred[i][pred_label[i]],2)
                label_freq = label_freq_dict[truth_label]
                if prob >= 0.9:
                    
                    total_freq[label_freq] += 1
                    total_freq['all'] += 1

                    correct_freq[label_freq] = correct_freq[label_freq] + 1 if pred_label[i] == label[i] else correct_freq[label_freq]
                    correct_freq['all'] = correct_freq['all'] + 1 if pred_label[i] == label[i] else correct_freq['all']
                    
                    semi_data.append([name, pseudo_label])

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
    pbar.close()
    print('save semi_data!!')
    np.savetxt(os.path.join(opt.data_path, 'val_semi.csv'), semi_data, fmt='%s', delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../dataset', help='path to data')
    parser.add_argument('--csv_path', default='../dataset/testcase', help='path to testing csv')
    parser.add_argument('--output_path', default='./pred', help='path to testing csv')
    parser.add_argument('--semi', action = 'store_true', help='Whether to generate psuedo label')

    parser.add_argument('--model', default='pvt_v2_b3', help='resnet50/resnet101/resnest50/resnest101/SEresnet50/SEresnet101/EfficientNet/pvt_v2_b2/pvt_v2_b3')
    parser.add_argument('--aug_ensemble', action = 'store_true', help='Whether to use five_crop')
    parser.add_argument('--val', action = 'store_true', help='validation mode or testing mode')
    parser.add_argument('--model_ensemble', action = 'store_true', help='Whether to ensemble')
    parser.add_argument('--load', default='./checkpoints/pvtv2_b3/reweight_pvt_v2_b3.pth', help='path to model to continue training')

    parser.add_argument('--image_size', type=int, default = 384, help='The size of image')
    parser.add_argument('--batch_size', type=int, default=16, help='number of cpu workers')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu workers')
    opt = parser.parse_args()
    os.makedirs(opt.output_path, exist_ok=True)

    if opt.val or opt.semi:
        test_data = food_data(opt.data_path, 'val', opt.image_size)
        test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, pin_memory=True)

    else:
        test_data = food_data(opt.data_path, 'test', opt.image_size, augment_ensemble=opt.aug_ensemble)
        test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, pin_memory=True)

    print('# of validation data: ', len(test_data))
    
    if opt.model_ensemble:
        model_1 = resnest50(num_classes=1000, pretrained=True).cuda()
        model_1.load_state_dict(torch.load('./model_epoch4_all79.49_f89.12_c75.60_r36.25.pth'))
        model_2 = resnest101(num_classes=1000, pretrained=True).cuda()
        model_2.load_state_dict(torch.load('./model_epoch0_all79.09_f87.66_c76.39_r36.33.pth'))
        model_3 = SEresnet50(num_classes=1000, pretrained=True).cuda()
        model_3.load_state_dict(torch.load('./model_epoch4_all77.84_f88.98_c72.76_r31.11.pth'))
        model_4 = SEresnet101(num_classes=1000, pretrained=True).cuda()
        model_4.load_state_dict(torch.load('./model_epoch7_all77.05_f83.15_c76.31_r39.93.pth'))
        model_5 = pvt_v2_b2(num_classes=1000).cuda()
        model_5 = torch.nn.DataParallel(model_5)
        model_5.load_state_dict(torch.load('./reweight_pvt_v2_b2.pth'))
        model_6 = pvt_v2_b3(num_classes=1000).cuda()
        model_6 = torch.nn.DataParallel(model_6)
        model_6.load_state_dict(torch.load('./reweight_pvt_v2_b3.pth'))
        model_7 = ArcSEresnet50(num_classes=1000, ArcFeature=512, pretrained=True).cuda()
        model_7.load_state_dict(torch.load('./model_epoch34_all76.81_f88.07_c71.79_r28.88.pth'))
        model_1.eval()
        model_2.eval()
        model_3.eval()
        model_4.eval()
        model_5.eval()
        model_6.eval()
        model_7.eval()
        model_list = [model_1, model_2, model_3, model_4, model_5, model_6, model_7]
        model = ensemble_net(model_list, opt.aug_ensemble).cuda()
    else:
        if opt.model == 'resnet50':
            model = resnet50(num_classes=1000, pretrained=True)
        elif opt.model == 'resnet101':
            model = resnet101(num_classes=1000, pretrained=True)
        elif opt.model == 'resnest50':
            model = resnest50(True, 1000)
        elif opt.model == 'resnest101':
            model = resnest101(True, 1000)
        elif opt.model == 'SEresnet50':
            model = SEresnet50(num_classes=1000, pretrained=True)
        elif opt.model == 'SEresnet101':
            model = SEresnet101(num_classes=1000, pretrained=True)
        elif opt.model == "EfficientNet":
            model = Efficinet_net(mode = opt.EfficientNet_mode, \
                                    advprop = False, \
                                    num_classes = opt.num_classes, \
                                    feature = opt.feature, \
                                    )
        elif opt.model == 'pvt_v2_b2':
            model = pvt_v2_b2(num_classes=1000)
            model = torch.nn.DataParallel(model)
        elif opt.model == 'pvt_v2_b3':
            model = pvt_v2_b3(num_classes=1000)
            model = torch.nn.DataParallel(model)
        
        model = model.cuda()
        model.load_state_dict(torch.load(opt.load))

    if opt.semi:
        semi_supervised(opt, model, test_loader)
    elif opt.val:
        val(opt, model, test_loader)
    else:
        if opt.aug_ensemble:
            print('augmentation ensemble testing!!')
            test_aug_ensemble(opt, model, test_loader)
        else:
            print('Standard testing!!')
            test(opt, model, test_loader)


    