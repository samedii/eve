import problem
import model
from torch.autograd import Variable
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


def main():
    toy_problem = problem.ToyProblem2()
    toy_prediction_model = model.ToyPredictionModel2()

    train(toy_problem, toy_prediction_model)

    test(toy_problem, toy_prediction_model)


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
            print([x for x in toy_prediction_model.named_parameters()])

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
