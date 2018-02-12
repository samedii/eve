import PredictionModel
import torch.nn as nn


class ToyPredictionModel(PredictionModel.PredictionModel):
    def __init__(self):
        super(PredictionModel.PredictionModel, self).__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer1(x)
