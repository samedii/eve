import torch.nn as nn
import torch.nn.functional as F


class ToyPredictionModel(nn.Module):
    def __init__(self):
        super(ToyPredictionModel, self).__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer1(x)


class ToyPredictionModel2(nn.Module):
    def __init__(self):
        super(ToyPredictionModel2, self).__init__()
        self.layer1 = nn.Linear(1, 3)
        self.layer2 = nn.Linear(3, 3)
        self.layer3 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x