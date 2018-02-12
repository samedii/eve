import torch.nn as nn


class PredictionModel(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, x):
        pass

class ToyPredictionModel(PredictionModel):
    def __init__(self):
        super(PredictionModel, self).__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer1(x)
