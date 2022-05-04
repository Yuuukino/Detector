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
from PerturbDataGenerate import AttackDataset
from torch.utils.data import Dataset, DataLoader
from AutoEncoder import AutoEncoder

class LatentDataset(Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        image = torch.tensor(image)
        label = torch.tensor(label)
        sample = {"Images": image, "Labels": label}
        return sample

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Generate_Latent_Space')
    parser.add_argument("--fgsm", action="store_true", default=False, help="FGSM Attack")
    parser.add_argument("--pgd", action="store_true", default=False, help="PGD Attack")
    parser.add_argument("--deepfool", action="store_true", default=False, help="Deep Fool Attack")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.ToTensor(), ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)


    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    model = AutoEncoder().to(device)
    models_dir = './models'
    file_name = '/AutoEncoder_CIFAR10.pth'
    model.load_state_dict(torch.load(models_dir + file_name))
    model.eval()

    df_name = []
    df_total = []

    if args.fgsm:
        fgsm = torch.load(models_dir + '/FGSM.pt')
        fgsm_ds = AttackDataset(fgsm['Images'],fgsm['Labels'])
        fgsm_dl = DataLoader(fgsm_ds, batch_size=100, shuffle=True)

        FGSM_Latent = []
        FGSM_Label = []
        
        for i, data in enumerate(fgsm_dl):
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            hidden, outputs = model(images)
            for j in range(len(hidden)):
                FGSM_Latent.append(hidden[j].cpu().detach().numpy())
                FGSM_Label.append(1)
            print('Iteration {}'.format(i))

        fgsm_images_labels_df = pd.DataFrame({'Images': FGSM_Latent, 'Labels': FGSM_Label})
        fgsm_dataset = LatentDataset(fgsm_images_labels_df['Images'],fgsm_images_labels_df['Labels'])
        fgsm_dataloader = DataLoader(fgsm_dataset, batch_size=2, shuffle=True)
    
        df_name.append('/FGSM_Latent.pt')
        df_total.append(fgsm_images_labels_df)
    
    if args.pgd:
        pgd = torch.load(models_dir + '/PGD.pt')
        pgd_ds = AttackDataset(pgd['Images'],pgd['Labels'])
        pgd_dl = DataLoader(pgd_ds, batch_size=100, shuffle=True)

        PGD_Latent = []
        PGD_Label = []
        
        for i, data in enumerate(pgd_dl):
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            hidden, outputs = model(images)
            for j in range(len(hidden)):
                PGD_Latent.append(hidden[j].cpu().detach().numpy())
                PGD_Label.append(1)
            print('Iteration {}'.format(i))

        pgd_images_labels_df = pd.DataFrame({'Images': PGD_Latent, 'Labels': PGD_Label})
        pgd_dataset = LatentDataset(pgd_images_labels_df['Images'],pgd_images_labels_df['Labels'])
        pgd_dataloader = DataLoader(pgd_dataset, batch_size=2, shuffle=True)

        df_name.append('/PGD_Latent.pt')
        df_total.append(pgd_images_labels_df)  
    
    if args.deepfool:
        deepfool = torch.load(models_dir + '/DeepFool.pt')
        deepfool_ds = AttackDataset(deepfool['Images'],deepfool['Labels'])
        deepfool_dl = DataLoader(deepfool_ds, batch_size=100, shuffle=True)

        deepfool_Latent = []
        deepfool_Label = []
        
        for i, data in enumerate(deepfool_dl):
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            hidden, outputs = model(images)
            for j in range(len(hidden)):
                deepfool_Latent.append(hidden[j].cpu().detach().numpy())
                deepfool_Label.append(1)
            print('Iteration {}'.format(i))

        deepfool_images_labels_df = pd.DataFrame({'Images': deepfool_Latent, 'Labels': deepfool_Label})
        deepfool_dataset = LatentDataset(deepfool_images_labels_df['Images'],deepfool_images_labels_df['Labels'])
        deepfool_dataloader = DataLoader(deepfool_dataset, batch_size=2, shuffle=True)
        df_name.append('/DeepFool_Latent.pt')
        df_total.append(deepfool_images_labels_df)  

    for i in range(len(df_name)):
        torch.save(df_total[i], models_dir + df_name[i])
        
    # for i in fgsm_dataloader:
    #     images = i["Images"]
    #     print(images)


