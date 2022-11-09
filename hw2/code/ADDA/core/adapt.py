"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn
from core.test import eval_tgt

import params
from utils import make_variable
import tqdm


def train_tgt(src_encoder, src_classifier, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, tgt_data_loader_eval):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################
    
    src_encoder.eval()
    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################
    best_acc = 0.
    for epoch in range(params.num_epochs):
        tgt_encoder.train()
        critic.train()
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        pbar = tqdm.tqdm(total=len_data_loader, ncols=0, desc="Train_adapt[%d/%d]"%(epoch, params.num_epochs), unit=" step")
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = images_src.cuda()
            images_tgt = images_tgt.cuda()

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.zeros(feat_src.shape[0]).type(torch.LongTensor).cuda()
            label_tgt = torch.ones(feat_tgt.shape[0]).type(torch.LongTensor).cuda()
            label_concat = torch.cat((label_src, label_tgt), 0).cuda()

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = torch.zeros(label_tgt.shape[0]).type(torch.LongTensor).cuda()

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            pbar.update()
            pbar.set_postfix(
            d_loss=f"{loss_critic.item():.4f}",
            g_loss=f"{loss_tgt.item():.4f}",
            domain_acc=f"{acc.item():.2f}",
            )
        pbar.close()
        #############################
        # 2.4 save model parameters #
        #############################
        torch.save(critic.state_dict(), os.path.join(params.model_root, "ADDA-critic-%d.pt"% epoch))
        torch.save(tgt_encoder.state_dict(), os.path.join(params.model_root, "ADDA-target-encoder-%d.pt"% epoch))
        
    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    return tgt_encoder
