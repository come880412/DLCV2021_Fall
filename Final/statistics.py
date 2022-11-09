import os
import operator
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import cv2

def Norm():
    # img_h, img_w = 32, 32
    img_h, img_w = 512, 512   #根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []
    
    train_path = '../dataset/train'
    train_folder_list = os.listdir(train_path)
    imgs_path_list = []
    for train_folder in train_folder_list:
        folder_path = os.path.join(train_path, train_folder)
        image_folder_name = os.listdir(folder_path)
        for image_name in image_folder_name:
            imgs_path = os.path.join(folder_path, image_name)
            imgs_path_list.append(imgs_path)
    
    len_ = len(imgs_path_list)
    i = 0
    for path in imgs_path_list:
        img = cv2.imread(path)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        if i%1000 == 0:
            print(i,'/',len_)    
    
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    
    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

def image_count():
    data_path = '../dataset/val'
    data_folder_list = os.listdir(data_path)

    count = {}
    
    for data_folder in tqdm.tqdm(data_folder_list):
        class_num = data_folder
        count[str(class_num)] = len(os.listdir(os.path.join(data_path, class_num)))
    count = dict( sorted(count.items(), key=operator.itemgetter(1),reverse=True))

    x_axis = np.arange(0, 1000, 1)
    y_axis = []
    for value in count.values():
        y_axis.append(value)
    y_axis = np.array(y_axis)

    plt.bar(x_axis, y_axis)
    plt.show() 

if __name__ == '__main__':
    Norm()
    # image_count()
    
