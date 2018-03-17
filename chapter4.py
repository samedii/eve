import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from sklearn.utils.extmath import cartesian

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

def main():

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()

class Net(nn.Module):
    layer_sizes = [(28*28, 10), (10, 10), (10, 10)]
    weights = None
    biases = None

    def __init__(self):
        super(Net, self).__init__()
        self.hyper_network = nn.Sequential(
            nn.Linear(5, 50),
            nn.ReLU(),
            torch.nn.BatchNorm1d(50),
            nn.Linear(50, 50),
            nn.ReLU(),
            torch.nn.BatchNorm1d(50),
            nn.Linear(50, 30),
            nn.ReLU(),
            torch.nn.BatchNorm1d(30),
            nn.Linear(30, 30),
            nn.ReLU(),
            torch.nn.BatchNorm1d(30),
            nn.Linear(30, 1),

            # 82%
            # nn.Linear(5, 50),
            # nn.ReLU(),
            # torch.nn.BatchNorm1d(50),
            # nn.Linear(50, 20),
            # nn.ReLU(),
            # torch.nn.BatchNorm1d(20),
            # nn.Linear(20, 20),
            # nn.ReLU(),
            # torch.nn.BatchNorm1d(20),
            # nn.Linear(20, 20),
            # nn.ReLU(),
            # torch.nn.BatchNorm1d(20),
            # nn.Linear(20, 20),
            # nn.ReLU(),
            # torch.nn.BatchNorm1d(20),
            # nn.Linear(20, 20),
            # nn.ReLU(),
            # torch.nn.BatchNorm1d(20),
            # nn.Linear(20, 1),
        )
        self.weight_data = [self.create_weight_data(layer_index) for layer_index in range(len(self.layer_sizes))]
        self.bias_data = [self.create_bias_data(layer_index) for layer_index in range(len(self.layer_sizes))]

    def calculate_layers(self):
        self.weights = [self.calculate_layer_weight(layer_index) for layer_index in range(len(self.layer_sizes))]
        self.biases = [self.calculate_layer_bias(layer_index) for layer_index in range(len(self.layer_sizes))]

    def hyper_forward(self, x):
        return self.hyper_network(x)

    def forward(self, x):
        if self.training:
            self.calculate_layers()

        x = x.view(-1, 28*28)
        for layer_index in range(len(self.layer_sizes)-1):
            x = x @ self.weights[layer_index] + self.biases[layer_index]
            x = F.relu(x)

        x = x @ self.weights[-1] + self.biases[-1]
        return F.log_softmax(x, dim=1)

    def calculate_layer_weight(self, layer_index):
        weight = self.hyper_forward(self.weight_data[layer_index])
        return weight.view(*self.layer_sizes[layer_index])

    def calculate_layer_bias(self, layer_index):
        bias = self.hyper_forward(self.bias_data[layer_index])
        return bias.view(self.layer_sizes[layer_index][1])

    def create_weight_data(self, layer_index):
        is_weight = np.asarray([1])
        layer = np.asarray([layer_index])
        layer_size = self.layer_sizes[layer_index]
        weight_row = np.arange(layer_size[0])
        weight_column = np.arange(layer_size[1])
        bias = np.asarray([0])
        weight_data = cartesian([is_weight, layer, weight_row, weight_column, bias])
        weight_data = weight_data.astype(np.float32)
        weight_data = torch.autograd.Variable(torch.from_numpy(weight_data))
        if args.cuda:
            weight_data = weight_data.cuda()
        return weight_data

    def create_bias_data(self, layer_index):
        is_weight = np.asarray([0])
        layer = np.asarray([layer_index])
        weight_row = np.asarray([0])
        weight_column = np.asarray([0])
        bias = np.arange(self.layer_sizes[layer_index][1])
        bias_data = cartesian([is_weight, layer, weight_row, weight_column, bias])
        bias_data = bias_data.astype(np.float32)
        bias_data = torch.autograd.Variable(torch.from_numpy(bias_data))
        if args.cuda:
            bias_data = bias_data.cuda()
        return bias_data

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()