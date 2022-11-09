import numpy as np
from torch.functional import norm
from dataset import p1_data
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
from timm import create_model
import os
from PIL import Image
import torch.nn.functional as F
import torch
import cv2
import tqdm
import sys
import matplotlib.pyplot as plt

def val(model, val_loader):
    model.eval()
    cuda = True if torch.cuda.is_available() else False
    label_total = 0
    correct_total = 0
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="val", unit=" step")
    for image, label, _ in val_loader:
        with torch.no_grad():
            if cuda:
                image = image.cuda()
                label = label.cuda()
            pred = model(image)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = (correct_total / label_total) * 100
            pbar.update()
            pbar.set_postfix(
            Accuracy=f"{acc:.2f}"
            )

def positional_embedding_visualization(model):
    pos_embed = model.module.pos_embed
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Visualization of position embedding similarities", fontsize=24)
    for i in tqdm.tqdm(range(1, pos_embed.shape[1])):
        sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
        sim = sim.reshape((14, 14)).detach().cpu()
        ax = fig.add_subplot(14, 14, i)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(sim)
    plt.savefig('./positional_embedding.png')
    plt.show()

def show_mask_on_image(img, attention_matrix):
    result = torch.eye(attention_matrix.size(-1))
    I = torch.eye(attention_matrix.size(-1)).unsqueeze(0)
    a = (attention_matrix + 1.0*I) / 2
    a = a / a.sum()

    result = torch.matmul(a, result)

    mask = result[0, 1:]
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    mask = cv2.resize(mask, img.size)

    img = np.float32(np.array(img)) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def Attention_map_visualization(model, val_loader):
    att = []
    for image, _, image_name in tqdm.tqdm(val_loader):
        with torch.no_grad():
            image = image.cuda()
            for i, img_name in enumerate(image_name):
                if img_name == '26_5064.jpg' or img_name =='29_4718.jpg' or img_name == '31_4838.jpg':
                    file_path = os.path.join('../../hw3_data/p1_data/val', img_name)
                    ori_img = Image.open(file_path)
                    ori_img = ori_img.resize((224,224))
                    fig = plt.figure(figsize=(16, 8))
                    fig.suptitle("Visualization of Attention", fontsize=24)
                    ax = fig.add_subplot(1, 2, 1)
                    ax.imshow(ori_img)
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    img = image[i].unsqueeze(0)
                    pred = model(img)
                    pred_label = torch.argmax(pred, dim=1)
                    patches = model.module.patch_embed(img)
                    pos_embed = model.module.pos_embed
                    transformer_input = torch.cat((model.module.cls_token, patches), dim=1) + pos_embed
                    x = transformer_input.clone()
                    for j, blk in enumerate(model.module.blocks):
                        if j == len(model.module.blocks) - 1: # The last layer of attention
                            # Attention Mechanism
                            norm = blk.norm1
                            attention = blk.attn
                            transformer_input_expanded = attention.qkv(norm(x))[0]
                            qkv = transformer_input_expanded.reshape(197, 3, 16, 64)  # (N=197, (qkv), H=16, D/H=64)
                            q = qkv[:, 0].permute(1, 0, 2)  # (H=16, N=197, D/H=64)
                            k = qkv[:, 1].permute(1, 0, 2)  # (H=16, N=197, D/H=64)
                            kT = k.permute(0, 2, 1)  # (H=16, D/H=64, N=197)
                            attention_matrix = q @ kT

                            
                            # Plot Attention map
                            attention_matrix = torch.mean(attention_matrix, dim=0).cpu()

                            # mask = attention_matrix[100, 1:].reshape((14, 14)).detach().numpy() # show attention_map
                            mask = show_mask_on_image(ori_img, attention_matrix.cpu()) # show image + attention_map
                            
                            ax = fig.add_subplot(1, 2, 2)
                            ax.imshow(mask)
                            ax.axes.get_xaxis().set_visible(False)
                            ax.axes.get_yaxis().set_visible(False)
                            plt.savefig('./attention_map/att_img_%s' % (img_name))
                            plt.show()
                        else:
                            x = blk(x)


if __name__ == '__main__':
    val_data_path = '../../hw3_data/p1_data/val'
    saved_model = './checkpoints/large/model_epoch8_acc95.73.pth'
    
    val_data = p1_data(val_data_path, 'val')
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)
    model = create_model('vit_large_patch16_224', pretrained=True)
    model.head = nn.Linear(in_features= model.head.in_features, out_features=37, bias=True)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(saved_model, map_location='cuda:0'))
    model.eval()
    
    # val(model, val_loader)
    # positional_embedding_visualization(model)
    Attention_map_visualization(model, val_loader)
