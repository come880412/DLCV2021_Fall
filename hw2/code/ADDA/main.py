"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed
from datasets.dataset import ADDA_data
from torch.utils.data import DataLoader
import torch
import os

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_train = ADDA_data(params.src_dataset, params.src_csv, 'train')
    src_data_val = ADDA_data(params.src_dataset, params.src_csv, 'val')
    tgt_data_train = ADDA_data(params.tgt_dataset, params.tgt_csv, 'train')
    tgt_data_valt = ADDA_data(params.tgt_dataset, params.tgt_csv, 'val')

    src_data_loader = DataLoader(dataset=src_data_train, batch_size=params.batch_size, shuffle=True, num_workers=params.n_cpu)
    src_data_loader_eval = DataLoader(dataset=src_data_val, batch_size=params.batch_size, shuffle=False, num_workers=params.n_cpu)
    tgt_data_loader = DataLoader(dataset=tgt_data_train, batch_size=params.batch_size, shuffle=True, num_workers=params.n_cpu)
    tgt_data_loader_eval = DataLoader(dataset=tgt_data_valt, batch_size=params.batch_size, shuffle=False, num_workers=params.n_cpu)

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

    # train source model
    print("=== Training classifier for source domain ===")
    # print(">>> Source Encoder <<<")
    # print(src_encoder)
    # print(">>> Source Classifier <<<")
    # print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader, src_data_loader_eval)
    # Load the best model during training
    src_encoder.load_state_dict(torch.load(os.path.join(params.model_root, 'ADDA-source-encoder-best.pt')))
    src_classifier.load_state_dict(torch.load(os.path.join(params.model_root, 'ADDA-source-classifier-best.pt')))

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    _ = eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    # print(">>> Target Encoder <<<")
    # print(tgt_encoder)
    # print(">>> Critic <<<")
    # print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, src_classifier, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader, tgt_data_loader_eval)

    tgt_encoder.load_state_dict(torch.load(os.path.join(params.model_root, 'ADDA-target-encoder-best.pt')))
    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    _ = eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    _ = eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
