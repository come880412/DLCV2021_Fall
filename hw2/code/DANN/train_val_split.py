import numpy as np
import os
import tqdm

def train_val(training_img_path, training_ratio, data):
    train_csv_file = np.loadtxt(training_img_path, delimiter=',', dtype=np.str)[1:]

    number_of_data = len(train_csv_file)
    random_num_list = np.random.choice(number_of_data, number_of_data, replace=False)

    train_index = np.array(random_num_list[:int(number_of_data * training_ratio)], dtype=int)
    val_index = np.array(random_num_list[int(number_of_data * training_ratio):], dtype=int)

    train_save_csv = []
    val_save_csv = []
    for i in tqdm.tqdm(range(number_of_data)):
        train_list = train_csv_file[i]
        if i in train_index:
            train_save_csv.append(train_list)
        elif i in val_index:
            val_save_csv.append(train_list)

    np.savetxt('./data_split/%s/train.csv' % data, train_save_csv, fmt='%s', delimiter=',')
    np.savetxt('./data_split/%s/val.csv' % data, val_save_csv, fmt='%s', delimiter=',')

if __name__ == '__main__':
    np.random.seed(500)
    data = 'usps' #mnistm, svhn, usps
    os.makedirs('./data_split/%s' % data, exist_ok= True)
    training_ratio = 0.8
    training_img_path = '../../hw2_data/digits/%s/train.csv' % data

    train_val(training_img_path, training_ratio, data)
    
