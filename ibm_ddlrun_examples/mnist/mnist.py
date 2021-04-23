"""
Based on https://github.com/xhzhao/PyTorch-MPI-DDP-example/blob/master/from_mingfei/mnist_dist.py

Synchronous SGD training on MNIST
Use distributed DDL backend

PyTorch distributed tutorial:
    http://pytorch.org/tutorials/intermediate/dist_tuto.html

This example make following updates upon the tutorial
1. Add params sync at beginning of each epoch
2. Allreduce gradients across ranks, not averaging
3. Sync the shuffled index during data partition
4. Remove torch.multiprocessing in __main__

Modifications:
*****************************************************************

Licensed Materials - Property of IBM

(C) Copyright IBM Corp. 2018. All Rights Reserved.

US Government Users Restricted Rights - Use, duplication or
disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

*****************************************************************

"""
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyddl
from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

gbatch_size = 128

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        """
        Be cautious about index shuffle, this is performed on each rank
        The shuffled index must be unique across all ranks
        Theoretically with the same seed Random() generates the same sequence
        This might not be true in rare cases
        You can add an additional synchronization for 'indexes', just for safety
        Anyway, this won't take too much time
        e.g.
            dist.broadcast(indexes, 0)
        """
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """
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


def partition_dataset(local_rank, rank):
    """ Partitioning MNIST """
    local_data = '../data_%s' % rank

    dataset = datasets.MNIST(
        local_data,
        train=True,
        download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    dist.barrier()
    bsz = int(gbatch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz

def sync_params(model):
    """ broadcast rank 0 parameter to all ranks """
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def sync_grads(model):
    """ all_reduce grads from all ranks """
    for param in model.parameters():
        dist.all_reduce(param.grad.data)

def sum_grads(model):
    """ all_reduce grads from all ranks """
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) 

def run(local_rank, rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset(local_rank, rank)
    device = torch.device("cuda:{}".format(local_rank))
    model = Net().to(device)
    model = model
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        # make sure we have the same parameters for all ranks
        sync_params(model)
        for data, target in train_set:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data.item()
            loss.backward()
            # all_reduce grads
            sum_grads(model)
            optimizer.step()
        print('Epoch {} Loss {:.6f} Global batch size {} on {} ranks'.format(
            epoch, epoch_loss / num_batches, gbatch_size, dist.get_world_size()))

def init_print(rank, size, debug_print=True):
    if not debug_print:
        if rank > 0:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
    else:
        # labelled print with info of [rank/size]
        old_out = sys.stdout
        class LabeledStdout:
            def __init__(self, rank, size):
                self._r = rank
                self._s = size
                self.flush = sys.stdout.flush

            def write(self, x):
                if x == '\n':
                    old_out.write(x)
                else:
                    old_out.write('[%d/%d] %s' % (self._r, self._s, x))

        sys.stdout = LabeledStdout(rank, size)

if __name__ == "__main__":
    # select ddl backend for allreduce
    dist.init_process_group(backend='ddl')
    # retrieve WORLD_SIZE & WORLD_RANK
    size = dist.get_world_size()
    rank = dist.get_rank()
    # retrieve local rank of process within node
    local_rank = pyddl.local_rank()
    init_print(rank, size)
    run(local_rank, rank, size)

