from network import Generator, Classifier
import argparse
import torch
import torch.nn.parallel
import os
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument('--img_size', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--latent_dim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--netG', default='./checkpoint/netG.pth', help="path to netG (to continue training)")
parser.add_argument('--single_sample_path', default='./single_sample', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1004, help='manual seed')
parser.add_argument('--n_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--sample_path', default='./test_result', help='folder to output images and model checkpoints')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)

opt = parser.parse_args()
os.makedirs(opt.sample_path, exist_ok=True)
os.makedirs(opt.single_sample_path, exist_ok=True)


netG = Generator(opt).cuda()
netG.load_state_dict(torch.load(opt.netG))
netG.eval()

netC = Classifier().cuda()
path = "Classifier.pth"
load_checkpoint(path, netC)

single_sample_images(opt, netG)
acc = Compute_Acc(opt, netC)
grid_sample_images(opt, netG, 0, n_row=10)
print('Classifer Acc: %.2f' % acc)