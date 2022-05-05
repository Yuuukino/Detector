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
from LatentSpace import LatentDataset
from torch.utils.data import Dataset, DataLoader

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(96 * 2 * 2, 192)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(192, 64)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 96 * 2 * 2)
        out = self.fc1(x)
        out = self.r1(out)
        out = self.fc2(out)
        out = self.r2(out)
        out = self.fc3(out)
        return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SimpleNet Train and Valid')
    parser.add_argument("--v", action="store_true", default=False, help="Perform validation only.")
    parser.add_argument("--train", default = "Train_AE_50000_15000_15000_20000.pt")
    parser.add_argument("--valid", default = "Valid_AE_50000_15000_15000_20000.pt")  
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models_dir = './models'
    batch_size = 100

    valid_pt = torch.load(models_dir + '/{}'.format(args.valid))
    valid_ds = LatentDataset(valid_pt['Images'],valid_pt['Labels'])
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

    train_pt = torch.load(models_dir + '/{}'.format(args.train))
    train_ds = LatentDataset(train_pt['Images'],train_pt['Labels'])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SimpleNet().to(device)
    models_dir = './models'
    file_name = '/SimpleNet.pth'

    if args.v:
        model.load_state_dict(torch.load(models_dir + file_name))
        model.eval()
        print("Start Testing!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in valid_dl:
                images = data["Images"]
                labels = data["Labels"]
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
    num_epochs = 20

    for epoch in range(num_epochs):
        for i, data in enumerate(train_dl):
            images = data["Images"]
            labels = data["Labels"]
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
        for data in valid_dl:
            images = data["Images"]
            labels = data["Labels"]
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





    

        