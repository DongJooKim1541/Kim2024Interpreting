''' Make data batches

This code contains modified parts of the original work(https://github.com/johnsk95/PT4AL)

Reference:
[1] Yi, J. S. K., Seo, M., Park, J. & Choi, D.-G. Pt4al: Using self-supervised pretext tasks for active learning. In European Conference on Computer Vision (Tel Aviv, Israel, 2022)

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np

from models import *
from loader import Loader, RotationLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = RotationLoader(is_train=False,  transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

net = ResNet18()
net.linear = nn.Linear(512, 4)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/rotation.pth')
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
            inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
            inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
            outputs = net(inputs)
            outputs1 = net(inputs1)
            outputs2 = net(inputs2)
            outputs3 = net(inputs3)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss4 = criterion(outputs3, targets3)
            loss = (loss1+loss2+loss3+loss4)/4.
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = loss.item()
            s = str(float(loss)) + '_' + str(path[0]) + "\n"

            with open('loss/rotation_loss.txt', 'a') as f:
                f.write(s)


if __name__ == "__main__":
    test()

    with open('loss/rotation_loss.txt', 'r') as f:
        losses = f.readlines()

    loss_1 = []
    name_2 = []

    for j in losses:
        loss_1.append(j[:-1].split('_')[0])
        name_2.append(j[:-1].split('_')[1])
    #print("len(name_2): ",len(name_2))
    s = np.array(loss_1)
    sort_index = np.argsort(s)
    x = sort_index.tolist()
    x.reverse()
    sort_index = np.array(x) # convert to high loss first
    sample5000=[]
    if not os.path.isdir('loss'):
        os.mkdir('loss')
    for i in range(10):
        # sample minibatch from unlabeled pool
        sample5000 = sort_index[i * 5000:(i + 1) * 5000]
        s = './loss/baseline_/batch_' + str(i) + '.txt'
        for k in sample5000:
            with open(s, 'a') as f:
                f.write(name_2[k]+'\n')
