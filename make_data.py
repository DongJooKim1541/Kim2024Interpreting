''' Make data

This code contains modified parts of the original work(https://github.com/johnsk95/PT4AL)

Reference:
[1] Yi, J. S. K., Seo, M., Park, J. & Choi, D.-G. Pt4al: Using self-supervised pretext tasks for active learning. In European Conference on Computer Vision (Tel Aviv, Israel, 2022)

'''

import torch
import torchvision
import os

class save_dataset(torch.utils.data.Dataset):

  def __init__(self, dataset, split='train'):
    self.dataset = dataset
    self.split = split

  def __getitem__(self, idx):
      data, label = self.dataset[idx]
      path = './DATA/'+self.split+'/'+str(label)+"/"+str(idx)+'.png'

      if not os.path.isdir('./DATA/'+self.split+'/'+str(label)):
          os.mkdir('./DATA/'+self.split+'/'+str(label))

      data.save(path)

  def __len__(self):
    return len(self.dataset)


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

train_dataset = save_dataset(trainset, split='train')
test_dataset = save_dataset(testset, split='test')

if not os.path.isdir('./DATA'):
    os.mkdir('./DATA')

if not os.path.isdir('./DATA/train'):
    os.mkdir('./DATA/train')

if not os.path.isdir('./DATA/test'):
    os.mkdir('./DATA/test')

for idx, i in enumerate(train_dataset):
    train_dataset[idx]

for idx, i in enumerate(test_dataset):
    test_dataset[idx]