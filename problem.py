import torch


class Problem:

    def loss(self, target, prediction):
        pass


class ToyProblem(Problem):
    def __init__(self):
        self.weight = 10
        self.bias = 2
        self.loss_function = torch.nn.MSELoss()

    def mean(self, data):
        return data*self.weight + self.bias

    def loss(self, target, prediction):
        return self.loss_function(prediction, target)

    @staticmethod
    def generate_data(n_samples):
        return torch.FloatTensor(n_samples, 1).normal_(0, 1)

    def generate_target(self, data):
        return self.mean(data) + torch.FloatTensor(data.size()).normal_(0, 0.1)


class ToyProblem2(ToyProblem):
    def mean(self, data):
        return data**2*self.weight + self.bias
