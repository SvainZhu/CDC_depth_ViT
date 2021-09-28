from __future__ import print_function, division
import torch.optim as optim
import sys
import time
import numpy as np
import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.utils import data
from models.utils import Dataset_Csv
import csv
from albumentations import *
from albumentations.pytorch import ToTensorV2
from models.statistic import calculate_statistic
from sklearn.metrics import roc_auc_score

model_type = 'ViT'
if model_type == 'ViT':
    from models.base_model import vit_base_patch16_224
# elif model_type == 'CDC_depth_ViT':
#     from models.cdc_depth_vit_model import vit_base_patch16_224
# elif model_type == 'CDC_ViT':
#     from models.cdc_vit_model import vit_base_patch16_224
else:
    raise ('Error model type!!!')


vit_transforms = Compose([
    Resize(224, 224),
    HorizontalFlip(p=0.5),
    HueSaturationValue(p=0.5),
    RandomBrightnessContrast(p=0.5),
    OneOf([
        IAAAdditiveGaussianNoise(),
        GaussNoise(),
    ], p=0.3),
    OneOf([
        MotionBlur(),
        GaussianBlur(),
        JpegCompression(quality_lower=65, quality_upper=80),
    ], p=0.5),
    Normalize([0.5] * 3, [0.5] * 3),
    ToTensorV2(),
]
)

def test_models(model, dataloaders, dataset_name):
    model.eval()
    y_scores, y_trues = [], []
    for k, (inputs_val, labels_val) in enumerate(dataloaders):
        inputs_val, labels_val = inputs_val.cuda(), labels_val.to(torch.float32).cuda()
        with torch.no_grad():
            outputs_val = model(inputs_val)
            outputs_val = outputs_val.squeeze(1)
            preds = torch.sigmoid(outputs_val)

        y_true = labels_val.data.cpu().numpy()
        y_score = preds.data.cpu().numpy()
        y_scores.extend(y_score)
        y_trues.extend(y_true)

    y_trues, y_scores = np.array(y_trues), np.array(y_scores)
    APCER, NPCER, ACER, ACC, HTER = calculate_statistic(y_scores, y_trues)
    AUC = roc_auc_score(y_trues, y_scores)
    print('\n===========Test Info===========\n')
    print(dataset_name, 'Test ACC: %5.4f' %(ACC))
    print(dataset_name, 'Test APCER: %5.4f' %(APCER))
    print(dataset_name, 'Test NPCER: %5.4f' %(NPCER))
    print(dataset_name, 'Test ACER: %5.4f' %(ACER))
    print(dataset_name, 'Test HTER: %5.4f' %(HTER))
    print(dataset_name, 'Test AUC: %5.4f' % (AUC))
    print('\n===============================\n')


def test_data(test_file):
    frame_reader = open(test_file, 'r')
    csv_reader = csv.reader(frame_reader)

    for f in csv_reader:
        img_path = f[0]
        label = int(f[1])
        test_label.append(label)
        test_list.append(img_path)



if __name__ == '__main__':
    # Modify the following directories to yourselves
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    batch_size = 16
    test_csv = r'H:/zsw/Data/CASIA_FASD/CSV/test.csv'      # The validation split file

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Data loading parameters
    params = {'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


    test_list = []
    test_label = []

    test_data(test_csv)

    test_set = Dataset_Csv(test_list, test_label, transform=vit_transforms)


    image_datasets = data.DataLoader(test_set, batch_size=batch_size, **params)

    model = vit_base_patch16_224(num_classes=1, has_logits=False)
    model.load_state_dict(torch.load('./model_out/ViT_CASIA_FASD/121999_ma.ckpt'))
    model = nn.DataParallel(model.cuda())
    dataset_name = "Oulu-Protocol1"

    test_models(model=model, dataloaders=image_datasets, dataset_name=dataset_name)
