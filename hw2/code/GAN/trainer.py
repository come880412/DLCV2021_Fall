import os
import time
import torch

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from Dataset import GAN_data
import tqdm
from cal_FID import main as FID_cal

from SAGAN import Generator_SAGAN, Discriminator_SAGAN
from utils import *

class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.use_tensorboard = config.use_tensorboard
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        # self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.max_IS = 0
        self.min_FID = 1000

        # Path
        # self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()



    def train(self):
        torch.manual_seed(412)
        criterion_GAN = GANLoss('wgangp').cuda()
        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(1000, self.z_dim))

        pbar = tqdm.tqdm(total=self.total_step, ncols=0, desc="Train", unit=" step")
        # Start with trained model
        start = 0

        # Start time
        real_label = torch.ones(self.batch_size)
        fake_label = torch.zeros(self.batch_size)
        combined_label = torch.cat((real_label, fake_label))
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
                real_images = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images = next(data_iter)

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = tensor2var(real_images)
            d_out_real,dr1,dr2 = self.D(real_images)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = criterion_GAN(d_out_real, True)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images,gf1,gf2 = self.G(z)
            d_out_fake,df1,df2 = self.D(fake_images.detach())

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = criterion_GAN(d_out_fake, False)
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            fake_pred = d_out_fake.cpu().detach()
            fake_pred = torch.where(fake_pred>0.5, 1, 0).squeeze()
            real_pred = d_out_real.cpu().detach()
            real_pred = torch.where(real_pred>0.5, 1, 0).squeeze()
            pred_combined = torch.cat((real_pred, fake_pred))
            pred_result = torch.eq(combined_label, pred_combined).sum()
            dis_acc = (pred_result / (self.batch_size * 2)) * 100
            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out,_,_ = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
            if step % self.d_iters == 0:
                # ================== Train G and gumbel ================== #
                # Create random noise
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
                fake_images,_,_ = self.G(z)

                # Compute loss with fake images
                g_out_fake,_,_ = self.D(fake_images)  # batch x n
                if self.adv_loss == 'wgan-gp':
                    g_loss_fake = criterion_GAN(g_out_fake, True)
                elif self.adv_loss == 'hinge':
                    g_loss_fake = - g_out_fake.mean()

                self.reset_grad()
                g_loss_fake.backward()
                self.g_optimizer.step()


            # Print out log info
            pbar.set_postfix(
            G_GAN=f"{g_loss_fake:.4f}",
            D_real=f"{d_loss_real:.4f}",
            D_fake=f"{d_loss_fake:.4f}",
            DIS_acc=f"{dis_acc:.2f}%",
            ave_gamma_l3=f"{self.G.attn1.gamma.mean().item():.4f}",
            ave_gamma_l4=f"{self.G.attn2.gamma.mean().item():.4f}",
            )

            pbar.update()
            self.writer.add_scalar('training_G_GAN', g_loss_fake, step)
            self.writer.add_scalar('training_D_real', d_loss_real, step)
            self.writer.add_scalar('training_D_fake', d_loss_fake, step)
            
            # Sample images
            if (step + 1) % self.sample_step == 0:
                self.G.eval()
                for idx, latent_vector in enumerate(fixed_z):
                    with torch.no_grad():
                        generated_img,_,_ = self.G(latent_vector.unsqueeze(0))
                        generated_img = denorm(generated_img.data)
                    # Display the generated image.
                        save_name = str(idx).zfill(4) + '.png'
                        save_image(generated_img, os.path.join(self.sample_path, save_name))
                generated_image_data = GAN_data(self.sample_path, 'test')
                IS, _ = inception_score(generated_image_data, cuda=True, resize=True)
                FID = FID_cal('../hw2_data/face/test', self.sample_path)
                print('\n FID:%.1f   IS:%.2f' % (FID, IS))
                # fake_images,_,_= self.G(fixed_z)
                # save_image(denorm(fake_images.data), os.path.join(self.sample_path, save_name))
                if self.max_IS < IS and FID < self.min_FID :
                    self.max_IS = IS
                    self.min_FID = FID
                    torch.save(self.G.state_dict(),
                            os.path.join(self.model_save_path, 'FIS%.1f_IS%.2f_G.pth' % (self.min_FID, self.max_IS)))
                    torch.save(self.D.state_dict(),
                            os.path.join(self.model_save_path, 'FIS%.1f_IS%.2f_D.pth' % (self.min_FID, self.max_IS)))
                    print('save model!!')
            # if (step+1) % model_save_step==0:
            #     torch.save(self.G.state_dict(),
            #                os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
            #     torch.save(self.D.state_dict(),
            #                os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def build_model(self):

        self.G = Generator_SAGAN(self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator_SAGAN(self.imsize, self.d_conv_dim).cuda()
        print(self.G)
        print(self.D)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter('runs/%s' % self.version)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))