import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import imageio
from torch import Tensor
from torch.autograd import Variable

Class = ['Urban','Agriculture','Rangeland','Forest','Water','Barren','Unknown']
Colormap = [[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]
Colormap_dict = {(0,255,255):0, (255,255,0):1, (255,0,255):2, (0,255,0):3, (0,0,255):4, (255,255,255):5, (0,0,0):6, (255,0,0):2,
                 0:(0,255,255), 1:(255,255,0), 2:(255,0,255), 3:(0,255,0), 4:(0,0,255), 5:(255,255,255), 6:(0,0,0)}

class CrossEntropy2d(nn.Module):

    def __init__(self, dim=1,  weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropy2d, self).__init__()

        """
        dim             : dimention along which log_softmax will be computed
        weight          : class balancing weight
        size_average    : which size average will be done or not
        ignore_index    : index that ignored while training
        """
        self.dim = dim
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        
        criterion =  nn.NLLLoss2d(self.weight, self.size_average, self.ignore_index)
        return criterion(F.log_softmax(input, dim=self.dim), target)


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def mask_to_label(mask):
    target = [ [ Colormap_dict[tuple(mask[:,i,j])] for j in range(mask.shape[2]) ] for i in range(mask.shape[1])]
    return np.array(target)

def pred_to_mask(image):
    b, h, w = image.shape
    output_img = np.zeros((b, h, w, 3))
    for batch in range(b):
        for height in range(h):
            for width in range(w):
                output_img[batch, height, width, :] = Colormap_dict[image[batch, height, width]]
    return output_img

class mean_iou_evaluate():
    def __init__(self):
        pass
    def read_masks(self, filepath):
        '''
        Read masks from directory and tranform to categorical
        '''
        file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
        file_list.sort()
        n_masks = len(file_list)
        masks = np.empty((n_masks, 512, 512))

        for i, file in enumerate(file_list):
            mask = imageio.imread(os.path.join(filepath, file))
            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
            masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
            masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
            masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
            masks[i, mask == 2] = 3  # (Green: 010) Forest land 
            masks[i, mask == 1] = 4  # (Blue: 001) Water 
            masks[i, mask == 7] = 5  # (White: 111) Barren land 
            masks[i, mask == 0] = 6  # (Black: 000) Unknown 

        return masks

    def mean_iou_score(self, pred, labels):
        '''
        Compute mean IoU score over 6 classes
        '''
        mean_iou = 0
        for i in range(6):
            tp_fp = np.sum(pred == i)
            tp_fn = np.sum(labels == i)
            tp = np.sum((pred == i) * (labels == i))
            iou = tp / (tp_fp + tp_fn - tp)
            mean_iou += iou / 6
        #     print('class #%d : %1.5f'%(i, iou))
        # print('\nmean_iou: %f\n' % mean_iou)

        return mean_iou
    
    def calculate(self, pred, labels):
        pred = self.read_masks(pred)
        labels = self.read_masks(labels)

        return self.mean_iou_score(pred, labels)

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

"""dice loss"""
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]