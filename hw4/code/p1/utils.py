import numpy as np
from torch.utils.data.sampler import Sampler
import pandas as pd
import torch
import csv
import sys
import torch.nn.functional as F

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            
class loss_metric():
    def __init__(self):
        pass
    def euclidean_metric(self, query_features, proto):
        n = query_features.shape[0]
        m = proto.shape[0]
        query_features = query_features.unsqueeze(1).expand(n, m, -1)
        proto = proto.unsqueeze(0).expand(n, m, -1)
        logits = -((query_features - proto)**2).sum(dim=2)
        return logits

    def cosine_similarity_metric(self, query_features, proto):
        n = query_features.shape[0]
        m = proto.shape[0]
        query_features = query_features.unsqueeze(1).expand(n, m, -1)
        proto = proto.unsqueeze(0).expand(n, m, -1)
        logits = F.cosine_similarity(query_features, proto, dim=2)
        return logits
    

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def compute_val_acc(gt_csv_path, output_csv_path):

    # read your prediction file
    with open(output_csv_path, mode='r') as pred:
        reader = csv.reader(pred)
        next(reader, None)  # skip the headers
        pred_dict = {int(rows[0]): np.array(rows[1:]).astype(int) for rows in reader}

    # read ground truth data
    with open(gt_csv_path, mode='r') as gt:
        reader = csv.reader(gt)
        next(reader, None)  # skip the headers
        gt_dict = {int(rows[0]): np.array(rows[1:]).astype(int) for rows in reader}

    if len(pred_dict) != len(gt_dict):
        sys.exit("Test case length mismatch.")

    episodic_acc = []
    for key, value in pred_dict.items():
        if key not in gt_dict:
            sys.exit("Episodic id mismatch: \"{}\" does not exist in the provided ground truth file.".format(key))

        episodic_acc.append((gt_dict[key] == value).mean().item())

    episodic_acc = np.array(episodic_acc)
    mean = episodic_acc.mean()
    std = episodic_acc.std()
    return mean * 100, 1.96 * std / (600)**(1/2) * 100

def euclidean_distance(query_features, proto):
    n = query_features.shape[0]
    m = proto.shape[0]
    query_features = query_features.unsqueeze(1).expand(n, m, -1)
    proto = proto.unsqueeze(0).expand(n, m, -1)
    logits = (query_features - proto)**2
    return logits