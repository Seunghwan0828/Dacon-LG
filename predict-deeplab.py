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
import math
import matplotlib.pyplot as plt
import zipfile
from datetime import datetime
import argparse
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from tqdm import tqdm

img_size = 256

# model_path  = argparse.ArgumentParser()
# model_path.add_argument('--pth', type=str, help="model.pth")
# print(model_path)

model_path = "deeplab-22_01_17-9.pth"


model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, aux_loss=None, pretrained_backbone=True)
model.classifier = DeepLabHead(in_channels=2048, num_classes=3)
model = model.to('cuda')
model.load_state_dict(torch.load(model_path))
model.eval()


# class UnNormalize(object):
#     def __init__(self,mean,std):
#         self.mean=[0.485, 0.456, 0.406]
#         self.std=[0.229, 0.224, 0.225]
 
#     def __call__(self,tensor):
#         """
#         Args:
#         :param tensor: tensor image of size (B,C,H,W) to be un-normalized
#         :return: UnNormalized image
#         """
#         for t, m, s in zip(tensor,self.mean,self.std):
#             t.mul_(s).add_(m)
#         return tensor


# img = Image.open('cat.jpg').convert('RGB')

# print(transforms.ToTensor()(img))
# # 0~1
# # [0.01, 0.0......]
# img_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# print(img_transforms(img))
# # -~+
# inv_normalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#     std=[1/0.229, 1/0.224, 1/0.225]
# )
# print(inv_normalize(img_transforms(img)))
# # 0~1 [0.01, 0.0, ....]


# sys.exit()



def predict(img_paths, stride=32, batch_size=8):
    results = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255  # 0~1
        crop = []
        position = []
        batch_count = 0

        result_img = np.zeros_like(img)
        voting_mask = np.zeros_like(img)

        for top in tqdm(range(0, img.shape[0], stride)):
            for left in range(0, img.shape[1], stride):
                piece = np.zeros([img_size, img_size, 3], np.float32)
                temp = img[top:top+img_size, left:left+img_size, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                crop.append(piece)
                position.append([top, left])
                batch_count += 1
                if batch_count == batch_size:
                    crop_np = np.array(crop)  # [256, 256, 3] -> [B, C, H, W]
                    # crop.transpose((2,0,1))
                    # crop.unsqueeze(0) - > [3, 256, 256] -> [1, 3, 256, 256]
                    crop = torch.tensor(crop_np)
                    crop = crop.permute(0, 3, 1, 2)
                    crop = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop)
                    crop = crop.to('cuda')
                    pred = model(crop)#*255
                    pred = pred['out']
                    crop = []
                    batch_count = 0
                    for num, (t, l) in enumerate(position):
                        # pred = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(pred)
                        # pred *= 255
                        # print(pred)
                        piece = pred[num]
                        piece = piece.cpu()
                        piece = piece.permute(1, 2, 0)
                        piece = piece.detach().numpy()
                        piece *= 255
                        h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
                        result_img[t:t+img_size, l:l+img_size, :] = np.add(result_img[t:t+img_size, l:l+img_size, :], piece[:h, :w], casting="unsafe")
                        voting_mask[t:t+img_size, l:l+img_size, :] += 1
                    position = []

        
        old_settings = np.seterr(all='ignore') 
        result_img = result_img/voting_mask
        result_img = result_img.astype(np.uint8)
        results.append(result_img)
        
    return results

def rmse_score(true, pred):
    score = math.sqrt(np.mean((true-pred)**2))
    return score

def psnr_score(true, pred, pixel_max):
    score = 20*np.log10(pixel_max/rmse_score(true, pred))
    return score

train_csv = pd.read_csv('./data/train.csv')
test_csv = pd.read_csv('./data/test.csv')
train_all_input_files = './data/train_input_img/' + train_csv['input_img']
train_all_label_files = './data/train_label_img/' + train_csv['label_img']

train_input_files = train_all_input_files[:].to_numpy()
train_label_files = train_all_label_files[:].to_numpy()

result = predict(train_input_files[:1], 32)

for i, (input_path, label_path) in enumerate(zip(train_input_files[:1], train_label_files[:1])):
    input_img = cv2.imread(input_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    targ_img = cv2.imread(label_path)
    targ_img = cv2.cvtColor(targ_img, cv2.COLOR_BGR2RGB)
    pred_img = result[i]
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(20,10))
    plt.subplot(1,3,1)
    plt.imshow(input_img)
    plt.title('input_img', fontsize=10)
    plt.subplot(1,3,2)
    plt.imshow(pred_img)
    plt.title('output_img', fontsize=10)
    plt.subplot(1,3,3)
    plt.imshow(targ_img)
    plt.title('target_img', fontsize=10)
    plt.show()
    print('input PSNR :', psnr_score(input_img.astype(float), targ_img.astype(float), 255))
    print('output PSNR :', psnr_score(result[i].astype(float), targ_img.astype(float), 255), '\n')
sys.exit()
test_input_files = './data/test_input_img/'+test_csv['input_img']
test_result = predict(test_input_files, 32)

for i, input_path in enumerate(test_input_files):
    input_img = cv2.imread(input_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    pred_img = test_result[i]
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(input_img)
plt.title('input_img', fontsize=10)
plt.subplot(1,2,2)
plt.imshow(pred_img)
plt.title('output_img', fontsize=10)
plt.show()

now = datetime.now()
date = now.strftime('%Y-%m-%d-%H:%M')
now_date = date

def make_submission(result):
    os.makedirs('submission', exist_ok=True)
    os.chdir("./submission/")
    sub_imgs = []
    for i, img in enumerate(result):
        path = f'test_{20000+i}.png'
        cv2.imwrite(path, img)
        sub_imgs.append(path)
    submission = zipfile.ZipFile("submission.zip", 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()

make_submission(test_result)