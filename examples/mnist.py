# 08-27-2019;
"""
MNIST training example.
Use `python mnist.py` to run with PyTorch NN.
Use `python mnist.py --warm` to run with PyWarm NN.
Use `python mnist.py --help` to see a list of cli argument options.
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('..')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import warm
import warm.functional as W


class WarmNet(nn.Module):
    def __init__(self):
        super().__init__()
        warm.up(self, [1, 1, 28, 28])
    def forward(self, x):
        x = W.conv(x, 20, 5, activation='relu')
        x = F.max_pool2d(x, 2)
        x = W.conv(x, 50, 5, activation='relu')
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 800)
        x = W.linear(x, 500, activation='relu')
        x = W.linear(x, 10)
        return F.log_softmax(x, dim=1)


class TorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(p, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx%p.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()))


def test(p, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    size = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= size
    print(f'\nTest loss: {test_loss:.4f}, Accuracy: {correct}/{size} ({100*correct/size:.2f}%)\n')


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--warm', action='store_true', help='use warm instead of vanilla pytorch.')
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument(
        '--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument(
        '--epochs', type=int, default=3, metavar='N', help='number of epochs to train (default: 3)')
    parser.add_argument(
        '--lr', type=float, default=0.02, metavar='LR', help='learning rate (default: 0.02)')
    parser.add_argument(
        '--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument(
        '--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N', help='number of batchs between logging training status')
    parser.add_argument(
        '--save-model', action='store_true', default=False, help='For Saving the current Model')
    p = parser.parse_args()
    #
    torch.manual_seed(p.seed)
    use_cuda = not p.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    kw = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=data_transform)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=p.batch_size, shuffle=True, **kw)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=p.test_batch_size, shuffle=True, **kw)
    model = WarmNet() if p.warm else TorchNet()
    print(f'Using {model._get_name()}.')
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=p.lr, momentum=p.momentum)
    print(f'Training with {p.epochs} epochs on {device} device.')
    #
    for i in range(p.epochs):
        train(p, model, device, train_loader, optimizer, i)
        test(p, model, device, test_loader)
    #
    if p.save_model:
        torch.save(model.state_dict(), 'mnist_cnn.pt')


if __name__ == '__main__':
    main()
