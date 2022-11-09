import torch
import torchvision
from utils import *
from config import get_parameters
from PIL import Image
from torchvision.utils import save_image
from cal_FID import main as FID_cal
import os
from Dataset import GAN_data
import random

from SAGAN import Generator_SAGAN

load_path = './checkpoints/hw2_face/SAGAN_model_G.pth'
num_output = 1000

torch.manual_seed(7414)
torch.cuda.manual_seed_all(7414)
np.random.seed(7414)
random.seed(7414)
torch.backends.cudnn.deterministic = True
os.makedirs('./generate_images/SAGAN', exist_ok=True)
save_path = './generate_images/SAGAN'

grid_image = torch.zeros((32, 3, 64, 64))
args = get_parameters()
# Create the generator network.
netG = Generator_SAGAN(args.imsize, args.z_dim, args.g_conv_dim).cuda()
# Load the trained generator weights.
netG.load_state_dict(torch.load(load_path))
netG.eval()

noise = torch.randn(num_output, args.z_dim).cuda()

# Turn off gradient calculation to speed up the process.
for idx, latent_vector in enumerate(noise):
	with torch.no_grad():
		generated_img,_,_ = netG(latent_vector.unsqueeze(0))
		generated_img = denorm(generated_img.data)
		if idx <32:
			grid_image[idx] = generated_img
		if idx == 32:
			save_image(torchvision.utils.make_grid(grid_image), './generate_images/sample.png')
	# Display the generated image.
		save_name = str(idx+1).zfill(4) + '.png'
		save_image(generated_img, os.path.join(save_path, save_name))

generated_image_data = GAN_data(save_path, 'test')
IS, _ = inception_score(generated_image_data, cuda=True, resize=True)
FID = FID_cal('../../hw2_data/face/test', save_path)
print('\n FID:%.1f   IS:%.2f' % (FID, IS))
