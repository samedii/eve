from torch.autograd import Variable
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


def main():
    toy_problem = ToyProblem2()
    toy_prediction_model = ToyPredictionModel2()

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


class ToyPredictionModel(nn.Module):
    def __init__(self):
        super(ToyPredictionModel, self).__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer1(x)


class ToyPredictionModel2(nn.Module):
    def __init__(self):
        super(ToyPredictionModel2, self).__init__()
        self.layer1 = nn.Linear(1, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layer3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x


def train(toy_problem, toy_prediction_model, batch_size=50, n_iterations=10000):

    optimizer = optim.SGD(toy_prediction_model.parameters(), lr=1e-3, momentum=1e-5)
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

        if iteration % 2 == 0:
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
