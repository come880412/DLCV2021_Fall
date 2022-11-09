from model.Resnet import ResNet, Bottleneck, SEResNeXt, SEBottleneckX101, SEBottleneck
from torchvision import models
import torch.nn as nn
import torch
from model.resnest.torch import resnest50, resnest101
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

def resnet50(num_classes, pretrained=False):
    model = ResNet(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes)
    if pretrained:
        dict_ = models.resnet50(pretrained = pretrained)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            if k in model_dict and k.split('.')[0] != 'fc':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet101(num_classes, pretrained=False):
    model = ResNet(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes)
    if pretrained:
        dict_ = models.resnet101(pretrained = pretrained)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            if k in model_dict and k.split('.')[0] != 'fc':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def SEresnet50(num_classes, pretrained=False):
    model = SEResNeXt(SEBottleneck, [3, 4, 6, 3], num_classes = num_classes)
    if pretrained:
        dict_ = models.resnext50_32x4d(pretrained = pretrained)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            if k in model_dict and k.split('.')[0] != 'fc':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def SEresnet101(num_classes, pretrained=False):
    model = SEResNeXt(SEBottleneckX101, [3, 4, 23, 3], num_classes = num_classes)
    if pretrained:
        dict_ = models.resnext101_32x8d(pretrained = pretrained)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            if k in model_dict and k.split('.')[0] != 'fc':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

class Efficinet_net(nn.Module):
    def __init__(self, mode, advprop, num_classes=1000, feature = 1280):
        super(Efficinet_net, self).__init__()
        self.efficientNet = EfficientNet.from_pretrained(mode, advprop=advprop)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature, num_classes)  # B0:1280, B5:2048, B6:2304


    def forward(self, x):
        self.efficientNet(x)
        features = self.efficientNet.extract_features(x)
        x = self.avgpool(features)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ensemble_net(nn.Module):
    def __init__(self, model_list, aug_ensemble):
        super(ensemble_net, self).__init__()
        self.model_list = model_list
        self.aug_ensemble = aug_ensemble

    """voting"""
    def forward(self, x):
        for idx, model in enumerate(self.model_list):
            if self.aug_ensemble:
                if idx == 0:
                    out = model(x)
                    out = torch.mean(out, dim=0)
                    out = out.unsqueeze(0)
                    out = F.softmax(out, dim=1)
                else:
                    pred = model(x)
                    pred = torch.mean(pred, dim=0)
                    pred = pred.unsqueeze(0)
                    pred = F.softmax(pred, dim=1)
                    out = torch.cat((out, pred), dim=0)
            else:
                if idx == 0:
                    out = F.softmax(model(x), dim=1)
                    out = out.unsqueeze(0)
                else:
                    pred = F.softmax(model(x), dim=1)
                    pred = pred.unsqueeze(0)
                    out = torch.cat((out, pred), dim=0)
        return torch.mean(out, dim=0)
