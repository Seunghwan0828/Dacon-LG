from PIL import Image
from torchvision.transforms import transforms
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import pandas as pd
import os
import numpy as np
import cv2
from functools import partial
from glob import glob
from multiprocessing import Pool, Process
from datetime import datetime
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision
from torch.cuda.amp import GradScaler, autocast, grad_scaler


img_size = 256

train_csv = pd.read_csv('./data/train.csv')
train_all_input_files = './data/train_input_img/' + train_csv['input_img']
train_all_label_files = './data/train_label_img/' + train_csv['label_img']



class CustomDataset(Dataset):
    def __init__(self, transform=None) -> None:
        super().__init__()
        # self.image_ids = glob('./data/train_image_256_128/*.jpg')
        # self.label_ids = glob('./data/train_label_256_128/*.jpg')
        self.image_ids = glob('./data/train_image_256/*.jpg')
        self.label_ids = glob('./data/train_label_256/*.jpg')
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        """
        returns:
            image FloatTensor([B, C, H, W]) - ~ + 
            label FloatTensor([B, C, H, W]) 0 ~ 1
        """
        image = Image.open(self.image_ids[index]).convert('RGB')
        label = Image.open(self.label_ids[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            label = transforms.ToTensor()(label)

        return image, label


def load_model():
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, aux_loss=None, pretrained_backbone=True)
    model.classifier = DeepLabHead(in_channels=2048, num_classes=3)
    model = model.to('cuda')

    return model


def main():
    model = load_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    scaler = GradScaler()


    

    train_transforms = transforms.Compose([
        # transforms.RandomAffine(30),
        # transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = CustomDataset(train_transforms)
    train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=6)

    print('Train Start!!')
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            # Size([B, 3, 256, 256]), Size([B, 1, 256, 256]) 0~1
            # [00000,0000,0000] [0000,0000,0000]
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                outputs = outputs['out']
                # print(outputs.size())
                loss = criterion(outputs, labels)

            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += outputs.shape[0] * loss.item()

            # print statistics
            running_loss += loss.item()
            if (i+1) % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # print epoch loss
        now = datetime.now()
        date = now.strftime('%d_%H_%M')
        print(epoch+1, epoch_loss / len(trainset))
        torch.save(model.state_dict(), f'./deeplab-{date}-{epoch}.pth')
        print(f'model{epoch} save!!!')


if __name__ == '__main__':
    main()