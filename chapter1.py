import problem
import model
from torch.autograd import Variable
import torch.optim as optim
import torch


def main():
    toy_problem = problem.ToyProblem()
    toy_prediction_model = model.ToyPredictionModel()

    train(toy_problem, toy_prediction_model)


def train(toy_problem, toy_prediction_model, batch_size=10, n_iterations=1000):

    optimizer = optim.SGD(toy_prediction_model.parameters(), lr=0.01, momentum=0.01)
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


if __name__ == '__main__':
    main()
