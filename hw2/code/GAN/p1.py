import torch
from utils import denorm
from torchvision.utils import save_image
import os
import sys
from SAGAN import Generator_SAGAN
import numpy as np
import random

# torch.manual_seed(7414)
torch.manual_seed(7414)
torch.cuda.manual_seed_all(7414)
np.random.seed(7414)
random.seed(7414)
torch.backends.cudnn.deterministic = True

save_path = sys.argv[1]
load_path = './model/p1/SAGAN_model_G.pth'

netG = Generator_SAGAN().cuda()
# Load the trained generator weights.
netG.load_state_dict(torch.load(load_path))
netG.eval()

noise = torch.randn(1000, 128).cuda()

for idx, latent_vector in enumerate(noise):
	with torch.no_grad():
		generated_img,_,_ = netG(latent_vector.unsqueeze(0))
		generated_img = denorm(generated_img.data)
		save_name = str(idx + 1).zfill(4) + '.png'
		save_image(generated_img, os.path.join(save_path, save_name))

