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
import concurrent.futures
from tqdm import tqdm
from PIL import Image
from numba import njit, prange
import numba


train_csv = pd.read_csv('./data/train.csv')
# test_csv = pd.read_csv('./data/test.csv')
# train_csv.head()
train_all_input_files = './data/train_input_img/' + train_csv['input_img']
train_all_label_files = './data/train_label_img/' + train_csv['label_img']
# test_all_label_files = './data/test_input_img/' + test_csv['input_img']

img_size = 256

os.makedirs('./data/train_image_256/', exist_ok=True)
os.makedirs('./data/train_label_256/', exist_ok=True)
# os.makedirs('./data/test_input_256_256/', exist_ok=True)

def cut_img(image_path, save_path, stride=256):
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    num = 0
    full_name = os.path.basename(image_path)
    name = os.path.splitext(full_name)[0]
    for top in range(0, img.shape[0], stride):
        for left in range(0, img.shape[1], stride):
            piece = np.zeros([img_size, img_size, 3], np.uint8)
            temp = img[top:top+img_size, left:left+img_size, :]
            piece[:temp.shape[0], :temp.shape[1], :] = temp
            num_str = str(num)
            num_str = num_str.zfill(3)
            cv2.imwrite(f'{save_path}/{name}-{num_str}.png', piece)
            # piece = Image.fromarray(piece)
            # piece.save(f'{save_path}/{name}-{num}.png')
            num += 1

    return


# list(tqdm(map(partial(cut_img, save_path='./data/train_image_256/'), train_all_input_files.to_list())))
# list(map(partial(cut_img, save_path='./data/train_label_256/'), train_all_label_files.to_list()))
# # sys.exit()

if __name__ == '__main__':
    # print(os.cpu_count()//2)
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor: # windows local 환경시 max_workers=os.cpu_count()//2
        list(tqdm(
            executor.map(partial(cut_img, save_path='./data/train_image_256/'), train_all_input_files.to_list()),             
            desc='train image cut',
            total=len(train_all_input_files)
        ))
        list(tqdm(
            executor.map(partial(cut_img, save_path='./data/train_label_256/'), train_all_label_files.to_list()),
            desc='train label image cut',
            total=len(train_all_input_files)
        ))
        # list(tqdm(
        #     executor.map(partial(cut_img, save_path='./data/test_input_256_256/'), test_all_label_files.to_list()),
        #     desc='test input image cut',
        #     total=len(test_all_label_files)
        # ))