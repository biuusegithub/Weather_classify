import torch
from torch import nn
from d2l import torch as d2l


def get_net_1():
    net = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(12544, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 4)
    )
    net = net.to(d2l.try_gpu())
    return net


def get_net_2():
    net = d2l.resnet18(4, 3)
    net = net.to(d2l.try_gpu())
    return net
