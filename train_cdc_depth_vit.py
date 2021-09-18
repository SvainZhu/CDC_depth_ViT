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
from sklearn.metrics import accuracy_score, roc_curve, auc
from random import shuffle
from models.utils_depth import Dataset_Csv
import csv
from sklearn.metrics import confusion_matrix
from models.cdc_depth_vit_model import vit_base_patch16_224, vit_base_patch32_224
import torch.nn.functional as F
from albumentations import *
from albumentations.pytorch import ToTensorV2

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


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float32)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)

    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1], input.shape[2])

    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth


class Contrast_depth_loss(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss, self).__init__()
        return

    def forward(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)

        criterion_MSE = nn.MSELoss().cuda()

        loss = criterion_MSE(contrast_out, contrast_label) * 0.001
        # loss = torch.pow(contrast_out - contrast_label, 2)
        # loss = torch.mean(loss)

        return loss


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def make_weights_for_balanced_classes(train_dataset, stage='train'):
    targets = []

    targets = torch.tensor(train_dataset)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight


def train_model(model, model_dir, criterion, depth_criterion, optimizer, scheduler, num_epochs=10, current_epoch=0):
    best_logloss = 10.0
    best_epoch = 0
    for epoch in range(current_epoch, num_epochs):
        best_test_logloss = 10.0
        epoch_start = time.time()
        model_out_path = os.path.join(model_dir, str(epoch) + '_vit.ckpt')
        log.write('------------------------------------------------------------------------\n')
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_loss_train = 0.0

            y_scores, y_trues = [], []
            for i, (inputs, maps_label, labels) in enumerate(dataloaders[phase]):
                inputs, maps_label, labels = inputs.cuda(), maps_label.to(torch.float32).cuda(), labels.to(torch.float32).cuda()

                if phase == 'train':
                    optimizer.zero_grad()
                    outputs, maps = model(inputs)
                    outputs = outputs.squeeze(1)
                    absolute_loss = criterion(outputs, labels)

                    absolute_loss.backward(retain_graph=True)

                    depth_loss = depth_criterion(maps, maps_label)
                    # cdc_module = ["map_concat", "map_extracter"]
                    for name, params in model.named_parameters():
                        if not "map_concat" in name:
                            params.requires_grad = False
                        if not "map_extracter" in name:
                            params.requires_grad = False
                    depth_loss.backward()

                    loss = absolute_loss + depth_loss
                    preds = torch.sigmoid(outputs)
                    optimizer.step()
                    for name, params in model.named_parameters():
                        params.requires_grad = True

                else:
                    with torch.no_grad():
                        outputs, maps = model(inputs)
                        outputs = outputs.squeeze(1)
                        absolute_loss = criterion(outputs, labels)
                        depth_loss = depth_criterion(maps, maps_label)
                        loss = absolute_loss + depth_loss
                        preds = torch.sigmoid(outputs)
                batch_loss = loss.data.item()
                running_loss += batch_loss
                running_loss_train += batch_loss

                y_true = labels.data.cpu().numpy()
                y_score = preds.data.cpu().numpy()

                if i % 100 == 0:
                    batch_acc = accuracy_score(y_true, np.where(y_score > 0.5, 1, 0))
                    log.write(
                        'Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(epoch,
                                                                                                      num_epochs - 1,
                                                                                                      i, len(
                                dataloaders[phase]), phase, batch_loss, batch_acc))
                if (i + 1) % 500 == 0:
                    inter_loss = running_loss_train / 1000.0
                    log.write('last phase train loss is {}\n'.format(inter_loss))
                    running_loss_train = 0.0
                    test_loss = val_models(model, criterion, depth_criterion, num_epochs, test_list, epoch)
                    if test_loss < best_test_logloss:
                        best_test_logloss = test_loss
                        log.write('save current model {}, Now time is {}, best logloss is {}\n'.format(i,time.asctime( time.localtime(time.time()) ),best_test_logloss))
                        model_out_paths = os.path.join(model_dir, str(epoch) + str(i) + '_vit.ckpt')
                        torch.save(model.module.state_dict(), model_out_paths)
                    model.train()
                    # scheduler.step()
                    log.write('now lr is : {}\n'.format(scheduler.get_lr()))

                if phase == 'test':
                    y_scores.extend(y_score)
                    y_trues.extend(y_true)
            if phase == 'test':
                epoch_loss = running_loss / (len(test_list) / batch_size)
                y_trues, y_scores = np.array(y_trues), np.array(y_scores)
                accuracy = accuracy_score(y_trues, np.where(y_scores > 0.5, 1, 0))

                log.write(
                    '**Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(epoch, num_epochs - 1, phase,
                                                                                        epoch_loss,
                                                                                        accuracy))
            if phase == 'test' and epoch_loss < best_logloss:
                best_logloss = epoch_loss
                best_epoch = epoch
                torch.save(model.module.state_dict(), model_out_path)
        log.write('Epoch {}/{} Time {}s\n'.format(epoch, num_epochs - 1, time.time() - epoch_start))
    log.write('***************************************************')
    log.write('Best logloss {:.4f} and Best Epoch is {}\n'.format(best_logloss, best_epoch))

def val_models(model, criterion, depth_criterion, num_epochs, test_list, current_epoch=0 ,phase='test'):
    log.write('------------------------------------------------------------------------\n')
    # Each epoch has a training and validation phase
    model.eval()
    running_loss_val = 0.0
    # print(phase)
    y_scores, y_trues = [], []
    for k, (inputs_val, maps_label, labels_val) in enumerate(dataloaders[phase]):
        inputs_val, maps_label, labels_val = inputs_val.cuda(), maps_label.to(torch.float32).cuda(), labels_val.to(torch.float32).cuda()
        with torch.no_grad():
            outputs_val, maps_val = model(inputs_val)
            outputs_val = outputs_val.squeeze(1)
            absolute_loss = criterion(outputs_val, labels_val)
            depth_loss = depth_criterion(maps_val, maps_label)
            loss = absolute_loss + depth_loss
            preds = torch.sigmoid(outputs_val)
        batch_loss = loss.data.item()
        running_loss_val += batch_loss

        y_true = labels_val.data.cpu().numpy()
        y_score = preds.data.cpu().numpy()

        if k % 100 == 0:
            batch_acc = accuracy_score(y_true, np.where(y_score > 0.5, 1, 0))
            log.write(
                'Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(current_epoch,
                                                                                              num_epochs - 1,
                                                                                              k, len(dataloaders[phase]),
                                                                                              phase, batch_loss, batch_acc))
        y_scores.extend(y_score)
        y_trues.extend(y_true)

    epoch_loss = running_loss_val / (len(test_list) / batch_size)
    y_trues, y_scores = np.array(y_trues), np.array(y_scores)
    accuracy = accuracy_score(y_trues, np.where(y_scores > 0.5, 1, 0))
    # torch.save(model.module.state_dict(), model_out_paths)
    log.write(
        '**Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(current_epoch, num_epochs - 1, phase,
                                                                            epoch_loss,
                                                                            accuracy))
    tn, fp, fn, tp = confusion_matrix(y_trues, np.where(y_scores > 0.5, 1, 0)).ravel()
    log.write(
        '**Epoch {}/{} Stage: {} TNR: {:.2f} FPR: {:.2f} FNR: {:.2f} TPR: {:.2f} \n'.format(current_epoch, num_epochs - 1, phase,
                                                                                            tn/(fp + tn),fp/(fp + tn),fn/(tp + fn),tp/(tp + fn)))
    log.write('***************************************************\n')
    # model.train()
    return epoch_loss

def base_data(train_file, train_map_file):
    frame_reader = open(train_file, 'r')
    csv_reader = csv.reader(frame_reader)

    for f in csv_reader:
        img_path = f[0]
        label = int(f[1])
        train_label.append(label)
        train_list.append(img_path)

    map_reader = open(train_map_file, 'r')
    csv_reader = csv.reader(map_reader)
    for f in csv_reader:
        map_path = f[0]
        train_map_list.append(map_path)

    log.write(str(len(train_list)) + '\n')

def validation_data(test_file, test_map_file):
    frame_reader = open(test_file, 'r')
    csv_reader = csv.reader(frame_reader)

    for f in csv_reader:
        img_path = f[0]
        label = int(f[1])
        test_label.append(label)
        test_list.append(img_path)

    map_reader = open(test_map_file, 'r')
    csv_reader = csv.reader(map_reader)
    for f in csv_reader:
        map_path = f[0]
        test_map_list.append(map_path)

    log.write(str(len(test_list)) + '\n')


if __name__ == '__main__':
    # Modify the following directories to yourselves
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.time()
    current_epoch = 10
    batch_size = 16
    train_csv = r'H:/zsw/Data/OULU/CSV/train_1.csv'  # The train split file
    val_csv = r'H:/zsw/Data/OULU/CSV/val_1.csv'      # The validation split file

    train_map_csv = r'H:/zsw/Data/OULU/CSV/train_map_1.csv'  # The train split file
    val_map_csv = r'H:/zsw/Data/OULU/CSV/val_map_1.csv'  # The validation split file

    #  Output path
    model_dir = 'E:/zsw/CDC_depth_ViT/model_out/CDC_depth_ViT1/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_name = model_dir.split('/')[-2] + '.log'
    log_dir = os.path.join(model_dir, log_name)
    if os.path.exists(log_dir):
        os.remove(log_dir)
        print('The log file is exit!')

    log = Logger(log_dir, sys.stdout)
    log.write('model : ViT   batch_size : 16 frames : 6 \n')
    log.write('pretrain : False   input_size : 224*224\n')

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Data loading parameters
    params = {'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_list = []
    train_map_list = []
    train_label = []
    log.write('loading train data' + '\n')
    base_data(train_csv, train_map_csv)

    ziplist = list(zip(train_list, train_map_list, train_label))
    shuffle(ziplist)
    train_list[:], train_map_list, train_label[:] = zip(*ziplist)

    test_list = []
    test_map_list = []
    test_label = []

    log.write('loading val data' + '\n')
    validation_data(val_csv, val_map_csv)

    train_set = Dataset_Csv(train_list, train_map_list, train_label, transform=vit_transforms)
    valid_set = Dataset_Csv(test_list, test_map_list, test_label, transform=vit_transforms)

    images_datasets = {}
    images_datasets['train'] = train_label
    images_datasets['test'] = test_label

    weights = {x: make_weights_for_balanced_classes(images_datasets[x], stage=x) for
               x in ['train', 'test']}
    data_sampler = {x: WeightedRandomSampler(weights[x], len(images_datasets[x]), replacement=True) for x in
                    ['train', 'test']}
    image_datasets = {}
    # over sampling
    image_datasets['train'] = data.DataLoader(train_set, sampler=data_sampler['train'], batch_size=batch_size, **params)

    # image_datasets['train'] = data.DataLoader(train_set, batch_size=batch_size, **params)
    image_datasets['test'] = data.DataLoader(valid_set, sampler=data_sampler['test'], batch_size=batch_size, **params)


    dataloaders = {x: image_datasets[x] for x in ['train', 'test']}
    datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    model = vit_base_patch16_224(num_classes=1, has_logits=False)
    model.train()
    model.load_state_dict(torch.load('./model_out/CDC_depth_ViT1/491999_vit.ckpt'))
    model = nn.DataParallel(model.cuda())


    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda()

    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)

    train_model(model=model, model_dir=model_dir, criterion=criterion,
                depth_criterion=criterion_contrastive_loss,
                optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler,
                num_epochs=60,
                current_epoch=current_epoch)

    elapsed = (time.time() - start)
    log.write('Total time is {}.\n'.format(elapsed))