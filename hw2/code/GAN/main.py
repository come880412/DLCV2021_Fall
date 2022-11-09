import torch
from Dataset import GAN_data
from trainer import Trainer
from torch.utils.data import DataLoader
from config import get_parameters
import os
from utils import *

if __name__ == "__main__":
    args = get_parameters()

    # Create directories if not exist
    make_folder(args.model_save_path, args.version)
    make_folder(args.sample_path, args.version)
    # make_folder(args.log_path, args.version)
    # make_folder(args.attn_path, args.version)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(412)
    train_data = GAN_data(args.root_train, 'train')
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    trainer = Trainer(train_loader, args)
    trainer.train()
    
