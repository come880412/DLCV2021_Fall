from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from network import Discriminator, Generator, weights_init_normal, Classifier
from dataset import ACGAN_data
from utils import Compute_Acc, set_requires_grad, grid_sample_images, single_sample_images, load_checkpoint
from torch.optim.lr_scheduler import LambdaLR
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--lr_decay", type=int, default=50, help="Start lr_decay")
parser.add_argument('--dataroot', default='../../hw2_data/digits/mnistm', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--img_size', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--latent_dim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--sample_path', default='./sample_result', help='folder to output images and model checkpoints')
parser.add_argument('--single_sample_path', default='./single_sample', help='folder to output images and model checkpoints')
parser.add_argument('--saved_model', default='./checkpoint', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1004, help='manual seed')
parser.add_argument('--n_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

opt = parser.parse_args()
print(opt)

# specify the gpu id if using only 1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
cuda = True if torch.cuda.is_available() else False

os.makedirs(opt.sample_path, exist_ok=True)
os.makedirs(opt.saved_model, exist_ok=True)
os.makedirs(opt.single_sample_path, exist_ok=True)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True

# datase t
dataset = ACGAN_data(opt.dataroot, 'train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

netC = Classifier()
path = "Classifier.pth"
load_checkpoint(path, netC)
# Define the generator and initialize the weights
netG = Generator(opt)
netG.apply(weights_init_normal)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

# Define the discriminator and initialize the weights
netD = Discriminator(opt)
netD.apply(weights_init_normal)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

print(netG)
print(netD)
# loss functions
adversarial_loss = torch.nn.BCEWithLogitsLoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# if using cuda
if cuda:
    netD.cuda()
    netG.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()
    netC.cuda()

# setup optimizer
optimizer_D = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_G = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

"""lr_scheduler"""
def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - opt.lr_decay) / float(opt.lr_decay + 1)
        return lr_l
scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D = LambdaLR(optimizer_D, lr_lambda=lambda_rule)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------
max_acc = 0.
for epoch in range(opt.n_epochs):
    pbar = tqdm.tqdm(total=len(dataloader), ncols=0, desc="Train[%d/%d]" % (epoch, opt.n_epochs), unit=" step")
    netG.train()
    netD.train()
    for i, data in enumerate(dataloader):
        imgs, labels = data
        imgs, labels = imgs.cuda().type(FloatTensor), labels.cuda().type(LongTensor)

        batch_size = imgs.shape[0]
        # Discriminator ground truths
        real_label = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake_label = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        set_requires_grad(netD, False)

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = netG(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        real_fake, pred_label = netD(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(real_fake, real_label) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        set_requires_grad(netD, True)

        # Loss for real images
        real_pred, real_aux = netD(imgs)
        d_real_loss = (adversarial_loss(real_pred, real_label) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = netD(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake_label) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator cls_accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_cls_acc = np.mean(np.argmax(pred, axis=1) == gt) * 100

        # Calculate discriminator domain_accuracy
        pred = np.concatenate([real_pred.data.cpu().numpy(), fake_pred.data.cpu().numpy()], axis=0)
        gt = np.concatenate([real_label.data.cpu().numpy(), fake_label.data.cpu().numpy()], axis=0)
        d_domain_acc = np.mean(np.argmax(pred, axis=1) == gt) * 100

        d_loss.backward()
        optimizer_D.step()

        pbar.update()
        pbar.set_postfix(
            loss_G=f"{g_loss.item():.4f}",
            loss_D=f"{d_loss.item():.4f}",
            D_domain_acc=f"{d_domain_acc:.2f}%",
            D_cls_acc=f"{d_cls_acc:.2f}%",
            cls_acc=f"{max_acc:.2f}%",
            )
    scheduler_D.step()
    scheduler_G.step()
    pbar.close()
    # do checkpointing
    grid_sample_images(opt, netG, epoch)
    single_sample_images(opt, netG)
    cls_acc = Compute_Acc(opt, netC)
    if cls_acc > max_acc:
        max_acc = cls_acc
        torch.save(netG.state_dict(), '%s/netG_epoch_%d_acc%.2f.pth' % (opt.saved_model, epoch, max_acc))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d_acc%.2f.pth' % (opt.saved_model, epoch, max_acc))
        print('saved_model!!')
