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
from torch.utils.data import Dataset, DataLoader
import foolbox as fb
import ResNet

class AttackDataset(Dataset):
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

    parser = argparse.ArgumentParser(description='Generate_perturbed_image')
    parser.add_argument("--fgsm", action="store_true", default=False, help="FGSM Attack")
    parser.add_argument("--pgd", action="store_true", default=False, help="PGD Attack")
    parser.add_argument("--deepfool", action="store_true", default=False, help="Deep Fool Attack")
    args = parser.parse_args()


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet.ResNet().to(device)
    models_dir = './models'
    file_name = '/ResNet_CIFAR10.pth'
    model.load_state_dict(torch.load(models_dir + file_name))
    model.eval()

    bounds = (-0.4914/0.1994, 1.4914/0.1994)
    fmodel = fb.PyTorchModel(model, bounds=bounds)

    df_name = []
    df_total = []

    if args.fgsm:
        FGSMimages = []
        FGSMlabels = []
        attack = fb.attacks.FGSM()
        total = 0
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons=0.3)
            
            total += labels.size(0)
            success = 0
            for i in range(labels.size(0)):
                if is_adv[i]:
                    FGSMimages.append(clipped[i].detach().cpu().numpy())
                    FGSMlabels.append(1)
                    success += 1
            print('Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / labels.size(0)))

        success = len(FGSMlabels)
        print('Summary: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / total))


        fgsm_images_labels_df = pd.DataFrame({'Images': FGSMimages, 'Labels': FGSMlabels})
        fgsm_dataset = AttackDataset(fgsm_images_labels_df['Images'],fgsm_images_labels_df['Labels'])
        fgsm_dataloader = DataLoader(fgsm_dataset, batch_size=2, shuffle=True)


        df_name.append('/FGSM.pt')
        df_total.append(fgsm_images_labels_df)
    
    if args.pgd:
        PGDimages = []
        PGDlabels = []
        attack = fb.attacks.PGD()
        total = 0
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons=0.3)
            
            total += labels.size(0)
            success = 0
            for i in range(labels.size(0)):
                if is_adv[i]:
                    PGDimages.append(clipped[i].detach().cpu().numpy())
                    PGDlabels.append(1)
                    success += 1
            print('Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / labels.size(0)))


        success = len(PGDlabels)
        print('Summary: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / total))


        pgd_images_labels_df = pd.DataFrame({'Images': PGDimages, 'Labels': PGDlabels})
        pgd_dataset = AttackDataset(pgd_images_labels_df['Images'],pgd_images_labels_df['Labels'])
        pgd_dataloader = DataLoader(pgd_dataset, batch_size=2, shuffle=True)

        df_name.append('/PGD.pt')
        df_total.append(pgd_images_labels_df)     
    
    if args.deepfool:
        DFimages = []
        DFlabels = []
        attack = fb.attacks.LinfDeepFoolAttack()
        total = 0
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons = 0.3)
            
            total += labels.size(0)
            success = 0
            for i in range(labels.size(0)):
                if is_adv[i]:
                    DFimages.append(clipped[i].detach().cpu().numpy())
                    DFlabels.append(1)
                    success += 1
            print('Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / labels.size(0)))

        success = len(DFlabels)
        print('Summary: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / total))


        df_images_labels_df = pd.DataFrame({'Images': DFimages, 'Labels': DFlabels})
        df_dataset = AttackDataset(df_images_labels_df['Images'],df_images_labels_df['Labels'])
        df_dataloader = DataLoader(df_dataset, batch_size=2, shuffle=True)

        df_name.append('/DeepFool.pt')
        df_total.append(df_images_labels_df)   


    for i in range(len(df_name)):
        torch.save(df_total[i], models_dir + df_name[i])

    # FF = torch.load(models_dir + '/FGSM.pt')
    # TD = AttackDataset(FF['Images'],FF['Labels'])
    # DL_DS = DataLoader(TD, batch_size=1, shuffle=True)
    # for i in DL_DS:
    #     images = i["Images"]
    #     print(images.shape)
    # print("Out")
