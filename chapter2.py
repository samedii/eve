from torch.autograd import Variable
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import cartesian
import torch.nn as nn
import torch.nn.functional as F


def main():
    toy_problem = ToyProblem2()
    toy_prediction_model = HyperModel()

    train(toy_problem, toy_prediction_model)

    test(toy_problem, toy_prediction_model)


class ToyProblem:
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


class HyperModel(nn.Module):
    layer_sizes = [(1, 5), (5, 5), (5, 1)]
    layer1_weight = None
    layer2_weight = None
    layer3_weight = None
    layer1_bias = None
    layer2_bias = None
    layer3_bias = None

    def __init__(self):
        super(HyperModel, self).__init__()
        self.hyper_layer1 = nn.Linear(5, 10)
        self.hyper_layer2 = nn.Linear(10, 10)
        self.hyper_layer3 = nn.Linear(10, 1)
        self.weight_data = [self.create_weight_data(layer_index) for layer_index in range(len(self.layer_sizes))]
        self.bias_data = [self.create_bias_data(layer_index) for layer_index in range(len(self.layer_sizes))]

    def calculate_layers(self):
        self.layer1_weight = self.calculate_layer_weight(0)
        self.layer2_weight = self.calculate_layer_weight(1)
        self.layer3_weight = self.calculate_layer_weight(2)
        self.layer1_bias = self.calculate_layer_bias(0)
        self.layer2_bias = self.calculate_layer_bias(1)
        self.layer3_bias = self.calculate_layer_bias(2)

    def forward(self, x):
        self.calculate_layers()  # probably want to cache this for batches

        x = x@self.layer1_weight + self.layer1_bias
        x = F.relu(x)
        x = x@self.layer2_weight + self.layer2_bias
        x = F.relu(x)
        x = x@self.layer3_weight + self.layer3_bias
        return x

    def calculate_layer_weight(self, layer_index):
        weight = self.hyper_layer1(self.weight_data[layer_index])
        weight = F.relu(weight)
        weight = self.hyper_layer2(weight)
        weight = F.relu(weight)
        weight = self.hyper_layer3(weight)
        weight = weight.view(*self.layer_sizes[layer_index])
        return weight

    def calculate_layer_bias(self, layer_index):
        bias = self.hyper_layer1(self.bias_data[layer_index])
        bias = F.relu(bias)
        bias = self.hyper_layer2(bias)
        bias = F.relu(bias)
        bias = self.hyper_layer3(bias)
        bias = bias.view(self.layer_sizes[layer_index][1])
        return bias

    def create_weight_data(self, layer_index):
        is_weight = np.asarray([1])
        layer = np.asarray([layer_index])
        layer_size = self.layer_sizes[layer_index]
        weight_row = np.arange(layer_size[0])
        weight_column = np.arange(layer_size[1])
        bias = np.asarray([0])
        weight_data = cartesian([is_weight, layer, weight_row, weight_column, bias])
        weight_data = weight_data.astype(np.float32)
        return torch.autograd.Variable(torch.from_numpy(weight_data))

    def create_bias_data(self, layer_index):
        is_weight = np.asarray([0])
        layer = np.asarray([layer_index])
        weight_row = np.asarray([0])
        weight_column = np.asarray([0])
        bias = np.arange(self.layer_sizes[layer_index][1])
        bias_data = cartesian([is_weight, layer, weight_row, weight_column, bias])
        bias_data = bias_data.astype(np.float32)
        return torch.autograd.Variable(torch.from_numpy(bias_data))


def train(toy_problem, toy_prediction_model, batch_size=100, n_iterations=20000):
    # doubled batch size from chapter1
    # doubled number of training iterations n_iterations
    # does not work every time but maybe 50%, final loss is around 2
    # worked with smaller networks too
    # sgd is sensitive to learning rate lr
    optimizer = optim.SGD(toy_prediction_model.parameters(), lr=1e-4, momentum=1e-1)
    toy_prediction_model.train()
    for iteration in range(n_iterations):
        data = toy_problem.generate_data(batch_size)
        target = toy_problem.generate_target(data)
        data, target = Variable(data), Variable(target)

        prediction = toy_prediction_model(data)
        loss = toy_problem.loss(target, prediction)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print('Iteration: {}\tLoss: {:.6f}'.format(
                iteration, torch.mean(loss.data)))
            # print([x for x in toy_prediction_model.named_parameters()])


def test(toy_problem, toy_prediction_model):

    n_samples = 100
    data = toy_problem.generate_data(n_samples)
    data, indices = data.sort(dim=0)

    target = toy_problem.generate_target(data)
    data, target = Variable(data), Variable(target)
    prediction = toy_prediction_model(data)

    plt.scatter(data.data.numpy(), target.data.numpy())
    plt.plot(data.data.numpy(), prediction.data.numpy())
    plt.show()


if __name__ == '__main__':
    main()
