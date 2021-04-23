'''
Based on https://github.ibm.com/mldlppc/pytorch-examples/blob/master-powerai/mnist/main.py

Modifications:

*****************************************************************

Licensed Materials - Property of IBM

(C) Copyright IBM Corp. 2018. All Rights Reserved.

US Government Users Restricted Rights - Use, duplication or
disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

*****************************************************************
'''

from __future__ import print_function
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import sys
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


def _get_train_data_loader(batch_size, training_dir, is_distributed, size, rank, download=False, **kwargs):
    logger.info("Get train data loader")

    dataset = datasets.MNIST(training_dir, train=True, download=download, transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, size, rank) if is_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs)

def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dist-backend', type=str, default='ddl',
                        help='backend for distributed training (mpi, ddl)')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    world_size, rank = None, None
    if args.dist_backend in ['mpi', 'ddl']:
        lrank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        dist.init_process_group(backend=args.dist_backend)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        args.distributed = True

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.distributed:
        torch.cuda.manual_seed(args.seed)

    if args.distributed:
        device = torch.device("cuda:{}".format(lrank) if use_cuda else "cpu")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'pin_memory': True} if use_cuda else {}
    data_dir = args.data_dir + "_" + str(rank)
    train_loader = _get_train_data_loader(args.batch_size, data_dir, args.distributed, world_size, rank, download=True, **kwargs)
    
    test_loader = _get_test_data_loader(args.test_batch_size, data_dir, **kwargs)

    model = Net().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        if (args.dist_backend in  ['mpi', 'ddl'] and int(lrank) == 0):
            torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    main()

