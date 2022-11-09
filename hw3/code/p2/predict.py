import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco
from configuration import Config
import matplotlib.pyplot as plt
import os
import cv2
import sys
import numpy as np


path = sys.argv[1]
save = sys.argv[2]
# path = '../../dataset/p2_data/images/'
# save = './attention_map/'

image_path_list = os.listdir(path)

config = Config()

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

@torch.no_grad()
def evaluate():
    count = 0
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions, att, feature1, feature2 = model(image, caption, cap_mask) # (B, max_position_embeddings, 30522)
        # print(att.shape)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            count += 1
            return caption, att, count, feature1, feature2

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
        count += 1

    return caption
model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token) # 101
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)   # 102

for image_name in image_path_list:
    image_path = os.path.join(path, image_name)
    image = Image.open(image_path)
    ori_img = image
    image = coco.val_transform(image)
    image = image.unsqueeze(0)
    ori_img = ori_img.resize((224,224))

    caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings)

    output, att, count, feature1, feature2 = evaluate()

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Visualization of Attention", fontsize=24)
    ax = fig.add_subplot(3, 5, 1)
    ax.imshow(ori_img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title('<start>')

    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    result_token = result.split(' ')
    result_token[-1] = result_token[-1][0:-1]
    for i in range(0,count-1, 1):
        if i >= len(result_token):
            token = '<end>'
        else:
            token = result_token[i]
        att_map = att[0, i, :]
        mask = att_map.reshape(feature1, feature2).numpy()
        mask = cv2.resize(mask, ori_img.size)
        ax = fig.add_subplot(3, 5, i+2)
        ax.imshow(mask, cmap='jet')
        ax.imshow(ori_img, alpha = 0.2)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(token)
    save_path = str(save) + str(image_name.split('.')[0] + '.png')
    plt.savefig(save_path)
    # plt.show()