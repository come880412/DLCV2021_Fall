"""Pre-train encoder and classifier for source dataset."""

from pathlib import WindowsPath
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model
import tqdm
import numpy as np
import torch


def train_src(encoder, classifier, src_data_loader, src_data_loader_eval):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    best_acc = 0.
    for epoch in range(params.num_epochs_pre):
        encoder.train()
        classifier.train()
        loss_total = 0.
        label_total = 0
        correct_total = 0
        pbar = tqdm.tqdm(total=len(src_data_loader), ncols=0, desc="Train_pretrain[%d/%d]"%(epoch, params.num_epochs_pre), unit=" step")
        for step, (images, labels) in enumerate(src_data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)
            loss_total += loss.item()

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # Calculate source acc
            pred = preds.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(labels.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            soruce_acc = (correct_total / label_total) * 100

            # print step info
            pbar.update()
            pbar.set_postfix(
            Source_train_loss=f"{loss_total:.4f}",
            Source_train_acc=f"{soruce_acc:.2f}%"
            )

        # eval model on test set
        pbar.close()
        test_acc = eval_src(encoder, classifier, src_data_loader_eval)

        # save model parameters
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(encoder, "ADDA-source-encoder-best.pt")
            save_model(classifier, "ADDA-source-classifier-best.pt")

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")
    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()
    label_total = 0
    correct_total = 0
    with torch.no_grad():
        # evaluate network
        for (images, labels) in data_loader:
            images = make_variable(images, volatile=True)
            labels = make_variable(labels)

            preds = classifier(encoder(images))
            loss += criterion(preds, labels).item()

            # Calculate source acc
            pred = preds.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(labels.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = correct_total / label_total

    loss /= len(data_loader)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    return acc
