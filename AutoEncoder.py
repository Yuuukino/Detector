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
import pandas as pd
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.fc1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        self.fc2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.fc3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 6, stride = 2)
        self.fc4 = nn.ReLU()

        self.convT1 = nn.ConvTranspose2d(in_channels = 96, out_channels = 64, kernel_size = 6, stride = 2)
        self.fc5 = nn.ReLU()
        self.convT2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        self.fc6 = nn.ReLU()
        self.convT3 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 4, stride = 2, padding = 1)
        self.fc7 = nn.ReLU()
        self.convT4 = nn.ConvTranspose2d(in_channels = 16, out_channels = 3, kernel_size = 3, stride = 1, padding = 1)
        self.last = nn.Sigmoid()

    def encode(self, x):
        out = self.conv1(x)
        out = self.fc1(out)
        out = self.conv2(out)
        out = self.fc2(out)
        out = self.conv3(out)
        out = self.fc3(out)
        out = self.conv4(out)
        out = self.fc4(out)

        return out
    
    def decode(self, z):
        out = self.convT1(z)
        out = self.fc5(out)
        out = self.convT2(out)
        out = self.fc6(out)
        out = self.convT3(out)
        out = self.fc7(out)
        out = self.convT4(out)
        
        return self.last(out)
    
    def forward(self, x):
        hidden = self.encode(x)
        out = self.decode(hidden)

        return hidden, out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AutoEncoder_CIFAR10_Pytorch')
    parser.add_argument("--valid", action="store_true", default=False, help="Perform validation only.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.ToTensor(), ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    model = AutoEncoder().to(device)
    models_dir = './models'
    file_name = '/AutoEncoder_CIFAR10.pth'

    if args.valid:
        model.load_state_dict(torch.load(models_dir + file_name))
        model.eval()
        print("Start Testing!")
        testiter = iter(testloader)
        images, labels = testiter.next()
        with torch.no_grad():
            plt.imshow(np.transpose(torchvision.utils.make_grid(images).cpu().numpy(), (1, 2, 0)))
            plt.show()
            hidden, outputs = model(images.to(device))
            plt.imshow(np.transpose(torchvision.utils.make_grid(outputs).cpu().numpy(), (1, 2, 0)))
            plt.show()
        exit(0)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 100

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)

            hiddens, outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100==0:
                print('Epoch {} Iteration {}, Loss: {}'.format(epoch, i, loss.data.cpu().numpy()))
    

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    torch.save(model.state_dict(), models_dir + file_name)