import config
from nets.resnext import *


class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        self.act = nn.ReLU(inplace=True)

        # Construct backbone network (ResNext-50), Global average pooling, Dropout
        self.ext = resnext50_32x4d()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)
   
        # BNNeck
        self.bnn = nn.BatchNorm2d(2048)
        nn.init.constant_(self.bnn.weight, 1)
        nn.init.constant_(self.bnn.bias, 0)
        self.bnn.bias.requires_grad_(False)
        
        # IDE
        self.fc_ide = nn.Linear(2048, config.num_ide_class, bias=False)
        
    def forward(self, patch):
        # Extract appearance feature
        feat_tri = self.avg_pool(self.ext(patch))

        # BNNeck
        feat_infer = self.bnn(self.drop(feat_tri))

        # IDE
        feat_ide = feat_infer.view(feat_infer.size(0), -1)
        ide = self.fc_ide(feat_ide)

        return feat_tri, feat_infer, ide
