import ToyProblem
import ToyPredictionModel
from torch.autograd import Variable
import torch.optim as optim
import torch


def main():
    toy_problem = ToyProblem.ToyProblem()
    toy_prediction_model = ToyPredictionModel.ToyPredictionModel()

    train(toy_problem, toy_prediction_model)


def train(problem, model, batch_size=10, n_iterations=1000):

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.01)
    model.train()
    for iteration in range(n_iterations):
        data = problem.generate_data(batch_size)
        target = problem.generate_target(data)
        data, target = Variable(data), Variable(target)

        prediction = model(data)
        loss = problem.loss(target, prediction)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 2 == 0:
            print('Iteration: {}\tLoss: {:.6f}'.format(
                iteration, torch.mean(loss.data)))
            print([x for x in model.named_parameters()])


if __name__ == '__main__':
    main()
