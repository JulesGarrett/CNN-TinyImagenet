import numpy as np
import os
from PIL import Image
import cv2
from tqdm import tqdm
import pprint as pp
import scipy
import csv
train_path = "/Users/julesgarrett/Downloads/tiny-imagenet-200/train"
label_path = "/Users/julesgarrett/Downloads/tiny-imagenet-200"
IMG_SIZE = 64

with open('wnids.txt') as file: #get relevant 200 ids
        ids = [line.rstrip('\n') for line in file]
ids = ids[0:10]

train_data =[]
labels = []
num_classes = 0
for file in tqdm(os.listdir(train_path)):
    if num_classes > 9:
        break
    if file in ids:
        for pic in os.listdir(train_path+"/"+file+"/images"):
            img = cv2.imread(train_path+"/"+file+"/images/"+pic)
            train_data.append(np.array(img))
            labels.append(file)
        num_classes += 1

train_data = np.array(train_data)
print(train_data.shape)
# np.save('tiny-imagenet-train.npy', train_data)


# labels = read.csv(label_path+"/"+"words.txt")
# labels = []
# with open(label_path+"/"+'words.txt') as f:
#     reader = csv.reader(f, delimiter = "\t")
#     labels = list(reader)
# np.save('tiny-imagenet-train-labels.npy', labels)
np.savez_compressed('extra-tiny-imagenet', train=train_data, labels=labels)
