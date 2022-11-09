"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
import numpy as np


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
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
            labels = make_variable(labels).squeeze_()

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
