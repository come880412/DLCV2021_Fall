"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed
from datasets.dataset import ADDA_data
from torch.utils.data import DataLoader
import torch
import os

data_root = '../../hw2_data/digits/'
src_dataset = 'svhn'  # mnistm/svhn/usps
tgt_dataset = 'mnistm' # usps/mnistm/svhn
model_root = './checkpoints_%s_%s' % (src_dataset, tgt_dataset)

# init random seed
init_random_seed(params.manual_seed)

# load dataset
src_data_test = ADDA_data(os.path.join(data_root, src_dataset, 'test'), os.path.join(data_root, src_dataset), 'test')
tgt_data_test = ADDA_data(os.path.join(data_root, tgt_dataset, 'test'), os.path.join(data_root, tgt_dataset), 'test')

src_data_loader_eval = DataLoader(dataset=src_data_test, batch_size=params.batch_size, shuffle=False, num_workers=params.n_cpu)
tgt_data_loader_eval = DataLoader(dataset=tgt_data_test, batch_size=params.batch_size, shuffle=False, num_workers=params.n_cpu)


# load models
src_encoder = init_model(net=LeNetEncoder(),
                            restore=params.src_encoder_restore)
src_classifier = init_model(net=LeNetClassifier(),
                            restore=params.src_classifier_restore)
tgt_encoder = init_model(net=LeNetEncoder(),
                            restore=params.tgt_encoder_restore)
critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                    hidden_dims=params.d_hidden_dims,
                                    output_dims=params.d_output_dims),
                    restore=params.d_model_restore)

tgt_encoder.load_state_dict(torch.load(os.path.join(model_root, 'ADDA-target-encoder-best.pt')))
src_encoder.load_state_dict(torch.load(os.path.join(model_root, 'ADDA-source-encoder-best.pt')))
src_classifier.load_state_dict(torch.load(os.path.join(model_root, 'ADDA-source-classifier-best.pt')))

print("=== Evaluating classifier for source domain ===")
_ = eval_src(src_encoder, src_classifier, src_data_loader_eval)
print(">>> source only <<<")
_ = eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
print(">>> domain adaption <<<")
_ = eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)