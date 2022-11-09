"""Main script for ADDA."""

import params
from models import LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed
from datasets.dataset import ADDA_data
from torch.utils.data import DataLoader
import torch
import csv
import numpy as np
import sys

def test(encoder, classifier, pred_path, tgt_data_loader_eval):
    encoder.eval()
    classifier.eval()
    csv_out = open(pred_path, 'w')
    csv_writer = csv.writer(csv_out, delimiter=',')
    csv_writer.writerow(['image_name', 'label'])
    idx = 0
    with torch.no_grad():
        for data in tgt_data_loader_eval:
            img, file_name_list  = data
            img = torch.FloatTensor(img)
            img = img.cuda()

            class_out = classifier(encoder(img))
            pred = class_out.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            for i in range(len(pred_label)):
                image_name = file_name_list[i]
                csv_writer.writerow([image_name, pred_label[i]])

data_path = sys.argv[1]
target = sys.argv[2]
pred_path = sys.argv[3]

data_root = '../hw2_data/digits/'
src_dataset = 'mnistm'  # mnistm/svhn/usps
tgt_dataset = 'usps' # usps/mnistm/svhn

# init random seed
init_random_seed(params.manual_seed)

# load dataset
tgt_data_test = ADDA_data(data_path, None, mode='savecsv')

tgt_data_loader_eval = DataLoader(dataset=tgt_data_test, batch_size=params.batch_size, shuffle=False, num_workers=params.n_cpu)


# load models
src_classifier = init_model(net=LeNetClassifier(),
                            restore=params.src_classifier_restore)
tgt_encoder = init_model(net=LeNetEncoder(),
                            restore=params.tgt_encoder_restore)
src_classifier, tgt_encoder, src_classifier.cuda(), tgt_encoder.cuda()
tgt_encoder.load_state_dict(torch.load('./model/bonus/%s/ADDA-target-encoder-best.pt'% target))
src_classifier.load_state_dict(torch.load('./model/bonus/%s/ADDA-source-classifier-best.pt'% target))

test(tgt_encoder, src_classifier, pred_path, tgt_data_loader_eval)