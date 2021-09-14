from __future__ import print_function, division
import os
import torch
from collections import Iterable
from torch.utils.data import DataLoader
from models.utils import Dataset_Csv
import csv
from models.cdc_depth_vit_model import vit_base_patch16_224, vit_base_patch32_224
from models.statistic import calculate_statistic
from sklearn.metrics import roc_auc_score
from albumentations import *
from albumentations.pytorch import ToTensorV2

img_transforms = Compose([
    Resize(224, 224),
    Normalize([0.5] * 3, [0.5] * 3),
    ToTensorV2()])

def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x


def test_data(csv_file):
    frame_reader = open(csv_file, 'r')
    fnames = csv.reader(frame_reader)
    for f in fnames:
        path = f[0]
        label = int(f[1])
        test_label.append(label)
        test_list.append(path)
    frame_reader.close()

def test_model(dataloaders, test_label, dataset_name):
    model = vit_base_patch16_224(num_classes=1, has_logits=False)
    model = model.cuda()
    model.load_state_dict(torch.load('./model_out/CDC_depth_ViT1/251499_vit.ckpt'))

    model.eval()
    preds_list = []
    with torch.no_grad():

        for i, (inputs, labels) in enumerate(dataloaders):
            inputs, labels = inputs.to(torch.float32).cuda(), labels.to(torch.float32).cuda()
            outputs, _ = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy().tolist()
            preds_list.append(preds)
    preds_list = list(flatten(preds_list))

    APCER, NPCER, ACER, ACC, HTER = calculate_statistic(preds_list, test_label)
    AUC = roc_auc_score(test_label, preds_list)
    print('\n===========Test Info===========\n')
    print(dataset_name, 'Test ACC: %5.4f' %(ACC))
    print(dataset_name, 'Test APCER: %5.4f' %(APCER))
    print(dataset_name, 'Test NPCER: %5.4f' %(NPCER))
    print(dataset_name, 'Test ACER: %5.4f' %(ACER))
    print(dataset_name, 'Test HTER: %5.4f' %(HTER))
    print(dataset_name, 'Test AUC: %5.4f' % (AUC))
    print('\n===============================\n')



if __name__ == "__main__":
    test_csv = r'H:/zsw/Data/OULU/CSV/test_1.csv'      # The test file dataset
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset_name = 'Oulu-Protocol1'
    batch_size = 16
    test_list = []
    test_label = []
    test_data(test_csv)
    test_set = Dataset_Csv(test_list, test_label, transform=img_transforms)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    test_model(test_dataloader, test_label, dataset_name)