from torch.autograd import Variable
from torch.serialization import save
from torch.utils import data
from torchvision.utils import save_image
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

import os
from dataset import cls_data
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def grid_sample_images(opt, generator, epoch, n_row=10):
    np.random.seed(opt.manualSeed)
    generator.eval()
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(10)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs, "%s/%d.png" % (opt.sample_path, epoch), nrow=10, normalize=True)


def single_sample_images(opt, generator):
    np.random.seed(opt.manualSeed)
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    generator.eval()
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (1000, opt.latent_dim))))
    labels = []
    # Get labels ranging from 0 to n_classes for n rows
    for label_num in range(10):
        for _ in range(100):
            labels.append(label_num)
    labels = np.array(labels)
    labels = Variable(LongTensor(labels))
    generated_img = generator(z, labels)
    for idx, saved_image in enumerate(generated_img):
        label = labels[idx].cpu().detach().numpy()
        save_name = str(label) + '_' + str((idx % 100) + 1).zfill(3) + '.png'
        save_image(saved_image.unsqueeze(0), os.path.join(opt.single_sample_path, save_name), normalize=True)
    

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


def Compute_Acc(opt, classifier):
    classifier.eval()
    dataset = cls_data(opt.single_sample_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=int(opt.workers))
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in data_loader:
            image, label = image.cuda().type(FloatTensor), label.cuda().type(LongTensor)
            pred = classifier(image)
            pred = pred.cpu().detach()
            pred_label = torch.argmax(pred, dim=1)
            correct += torch.eq(pred_label, label.cpu()).sum()
            total += len(label)
    acc = (correct / total) * 100
    return acc
    

