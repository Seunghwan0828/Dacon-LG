import sys
from tqdm import tqdm
import os


image_path = './data/train_image_256'
image_ids = os.listdir(image_path)
for name in tqdm(image_ids):
    src = os.path.join(image_path, name)
    rename = src[:-3]
    dst = rename + 'jpg'
    os.rename(src, dst)



label_path = './data/train_label_256'
label_ids = os.listdir(label_path)
for name in tqdm(label_ids):
    src = os.path.join(label_path, name)
    rename = src[:-3]
    dst = rename + 'jpg'
    os.rename(src, dst)

