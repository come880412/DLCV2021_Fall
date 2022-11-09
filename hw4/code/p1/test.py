from torch.utils.data import DataLoader
from dataset import MiniDataset_test, MiniImageNet_train
from utils import worker_init_fn, GeneratorSampler, CategoriesSampler, count_acc, compute_val_acc, loss_metric
from model import Convnet
import argparse
import torch
import numpy as np
import torch.nn.functional as F



def test(args, model, valid_loader):
    model.eval()
    gt_csv = np.loadtxt(args.valcase_csv, delimiter=',', dtype=np.str)
    save_data = np.concatenate((gt_csv[:,0][:,np.newaxis], gt_csv[:,6:]), axis=1)
    total_loss = 0.
    metric = loss_metric()
    with torch.no_grad():
        for i, (data, target) in enumerate(valid_loader):
            support_input = data[:5 * args.N_shot,:,:,:].cuda()
            query_input   = data[5 * args.N_shot:,:,:,:].cuda()

            label_encoder = {target[i * args.N_shot] : i for i in range(5)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[5 * args.N_shot:]])
            
            proto = model(support_input)
            proto = proto.reshape(args.N_shot, 5, -1).mean(dim=0)

            query_features = model(query_input)
            
            logits = metric.euclidean_metric(query_features, proto)
            # logits = metric.cosine_similarity_metric(query_features, proto)
            loss = F.cross_entropy(logits, query_label)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1).cpu().detach().numpy()
            pred = pred.astype(str)

            save_data[i+1,1:] = pred
    np.savetxt(args.output_csv, save_data, fmt='%s', delimiter=',')

    # val_acc, error = compute_val_acc(args.valcase_gt_csv, args.output_csv)
    # print('Accuracy: {:.2f} +- {:.2f} %'.format(val_acc, error))

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of epochs for training')
    parser.add_argument('--load', type=str, default='./model_p1.pth', help="Model checkpoint path")
    parser.add_argument('val_csv', type=str, default='../../hw4_data/mini/val.csv', help="Validation images csv file")
    parser.add_argument('val_data_dir', type=str, default='../../hw4_data/mini/val', help="Validation images directory")
    parser.add_argument('valcase_csv', type=str, default='../../hw4_data/mini/val_testcase.csv', help="Validation case csv")
    parser.add_argument('--valcase_gt_csv', type=str, default='./hw4_data/mini/val_testcase_gt.csv', help="Validation case csv")
    parser.add_argument('output_csv', type=str, default='./predict.csv', help="Output filename")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    val_dataset = MiniDataset_test(args.val_csv, args.val_data_dir)

    val_loader = DataLoader(
        val_dataset, batch_size= 5 * (args.N_query + args.N_shot),
        num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.valcase_csv))

    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))

    test(args, model, val_loader)