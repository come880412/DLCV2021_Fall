import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from pytorch_grad_cam import GradCAM

def label_to_freq(pred_label, label, label_freq_dict, total_freq, correct_freq):
    for i in range(len(pred_label)):
        pred_freq = label_freq_dict[pred_label[i]]
        label_freq = label_freq_dict[label[i]]
        total_freq[label_freq] += 1
        total_freq['all'] += 1

        correct_freq[label_freq] = correct_freq[label_freq] + 1 if pred_label[i] == label[i] else correct_freq[label_freq]
        correct_freq['all'] = correct_freq['all'] + 1 if pred_label[i] == label[i] else correct_freq['all']
    return total_freq, correct_freq

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s = 30.0, m = 0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class FocalLoss(nn.Module):

    def __init__(self, num_classes, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        # self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_classes, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_classes)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_classes:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask
        # ----------memory saving way--------
        prob = prob.gather(1, target).view(-1) + 1e-9
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.view(-1))
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class Balanced_CE(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(Balanced_CE, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def grad_cam(input_tensor, model):
    target_layers = [model.layer4[-1]]
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, aug_smooth=True)

    return grayscale_cam