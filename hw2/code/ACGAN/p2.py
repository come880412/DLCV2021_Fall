from network import Generator, Classifier
import argparse
import torch
import torch.nn.parallel
import os
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import sys

def single_sample_images(saved_path, generator):
    np.random.seed(1004)
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    generator.eval()
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (1000, 100))))
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
        save_image(saved_image.unsqueeze(0), os.path.join(saved_path, save_name), normalize=True)

parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument('--img_size', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--latent_dim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--netG', default='./model/p2/netG.pth', help="path to netG (to continue training)")
parser.add_argument('--single_sample_path', default='./single_sample', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1004, help='manual seed')
parser.add_argument('--n_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--sample_path', default='./test_result', help='folder to output images and model checkpoints')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('saved_path', type=str, nargs=1,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

opt = parser.parse_args()
saved_path = opt.saved_path[0]
netG = Generator(opt).cuda()
netG.load_state_dict(torch.load(opt.netG))
netG.eval()
single_sample_images(saved_path, netG)
