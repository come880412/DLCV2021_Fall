import torch.nn as nn
from torch.autograd import Function

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        self.label_pred = nn.Sequential(
            nn.Linear(64 *4 *4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(64 *4 *4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
        )

    def forward(self, x, domain=False , alpha=None):
        x = self.feature_extractor(x)
        feature = x.view(-1, 64* 4* 4) # flatten
        class_out = self.label_pred(feature)
        if domain:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            return class_out, domain_output
        else:
            return class_out

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
