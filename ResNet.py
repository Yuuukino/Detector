import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, identity):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if identity:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride, bias = False)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)

        self.block1 = self._make_layers(16, 16, 1, True)
        self.block2 = self._make_layers(16, 32, 2, False)
        self.block3 = self._make_layers(32, 64, 2, False)
        
        self.last = nn.Linear(64 * 2 * 2, num_classes)
    
    def _make_layers(self, in_channels, out_channels, stride, identity):
        return nn.Sequential(ResBlock(in_channels, out_channels, stride, identity),
                             ResBlock(out_channels, out_channels, 1, False))
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        #print(out.shape)
        out = self.block1(out)
        #print(out.shape)
        out = self.block2(out)
        #print(out.shape)
        out = self.block3(out)
        #print(out.shape)
        out = F.avg_pool2d(out, 4)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.last(out)

        return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ResNet_CIFAR10_Pytorch')
    parser.add_argument("--valid", action="store_true", default=False, help="Perform validation only.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    model = ResNet().to(device)
    models_dir = './models'
    file_name = '/ResNet_CIFAR10.pth'

    if args.valid:
        model.load_state_dict(torch.load(models_dir + file_name))
        model.eval()
        print("Start Testing!")
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, result = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (result == labels).sum().item()
        print('Total: {}, Correct: {}, Percentage: {}%'.format(total, correct, correct * 100 / total))

        exit(0)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 200

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100==0:
                print('Epoch {} Iteration {}, Loss: {}'.format(epoch, i, loss.data.cpu().numpy()))
    
    model.eval()
    print("Start Testing!")
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, result = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (result == labels).sum().item()
    print('Total: {}, Correct: {}, Percentage: {}%'.format(total, correct, correct * 100 / total))


    

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    torch.save(model.state_dict(), models_dir + file_name)