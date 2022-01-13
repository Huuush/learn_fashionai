import timm
import torch.nn as nn
from main import CFG


class myModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(myModel, self).__init__()

        self.feature_extractor = timm.create_model(CFG.model_name, pretrained=True, features_only=True)
        self.act = nn.Sigmoid()
        self.classfier = nn.Linear()

    def forward(self, x):
        self.feature_extractor(x)

