import numpy
from PIL import Image
from torch.utils import data
import torch
import cv2
import math

face_scale = 1.6

## ---------------------- Dataloaders ---------------------- ##
class Dataset_Csv(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, img_root, map_root, labels, transform=None):
        "Initialization"
        # self.data_path = data_path
        self.labels = labels
        self.img_root = img_root
        self.map_root = map_root
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.img_root)




    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample

        img_path = self.img_root[index]
        bbox_path = img_path.split('.')[0] + '.dat'
        map_path = self.map_root[index]

        # Load data
        img = cv2.imread(img_path)
        face_temp = crop_face_from_scene(image=img, face_name_full=bbox_path, scale=face_scale)
        face = self.transform(image=face_temp)['image']
        map_temp = cv2.imread(map_path, 0)
        map = cv2.resize(crop_face_from_scene(image=map_temp, face_name_full=bbox_path, scale=face_scale), (32, 32))
        label = self.labels[index]  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return face, map, label

def crop_face_from_scene(image, face_name_full, scale):
    f = open(face_name_full, 'r')
    lines = f.readlines()
    y1, x1, w, h = [float(ele) for ele in lines[:4]]
    f.close()
    y2 = y1 + w
    x2 = x1 + h

    y_mid = (y1 + y2) / 2.0
    x_mid = (x1 + x2) / 2.0
    h_img, w_img = image.shape[0], image.shape[1]
    # w_img,h_img=image.size
    w_scale = scale * w
    h_scale = scale * h
    y1 = y_mid - w_scale / 2.0
    x1 = x_mid - h_scale / 2.0
    y2 = y_mid + w_scale / 2.0
    x2 = x_mid + h_scale / 2.0
    y1 = max(math.floor(y1), 0)
    x1 = max(math.floor(x1), 0)
    y2 = min(math.floor(y2), w_img)
    x2 = min(math.floor(x2), h_img)

    # region=image[y1:y2,x1:x2]
    if image.size == 3:
        region = image[x1:x2, y1:y2, :]
    else:
        region = image[x1:x2, y1:y2]
    return region

## ---------------------- end of Dataloaders ---------------------- ##



