import os
import csv
import json
import random
import pandas as pd

interval = 6

def base_process(image_dir, map_dir, image_csv, map_csv):
    map_csv_a = open(map_csv, 'a', encoding='utf-8', newline='')
    map_csv_writer = csv.writer(map_csv_a)
    with open(image_csv, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in os.listdir(map_dir):
            file_path1 = os.path.join(map_dir, i)
            if i == 'attack_face':
                label = 0
            else:
                label = 1
            for j in os.listdir(file_path1):
                file_path2 = os.path.join(file_path1, j)
                for k in os.listdir(file_path2):
                    count = 0
                    # if count % interval != 0:
                    #     continue
                    face_name = '_'.join(k.split('_')[:-1]) + '.jpg'
                    map_name = k
                    map_path = os.path.join(file_path2, map_name)
                    image_path = os.path.join(os.path.join(image_dir, i), j)
                    face_path = os.path.join(image_path, face_name)
                    csv_writer.writerow([face_path, label])
                    map_csv_writer.writerow([map_path, label])
                else:
                    continue
    map_csv_a.close()

def Oulu_process():
    Protocol = '4'

    train_image_dir = 'H:/zsw/Data/OULU/Image/Train_files/'
    val_image_dir = 'H:/zsw/Data/OULU/Image/Dev_files/'
    test_image_dir = 'H:/zsw/Data/OULU/Image/Test_files/'

    train_map_dir = 'H:/zsw/Data/OULU/Depth_Map/Train_files/'
    val_map_dir = 'H:/zsw/Data/OULU/Depth_Map/Dev_files/'
    test_map_dir = 'H:/zsw/Data/OULU/Depth_Map/Test_files/'

    train_list = 'D:/Data/OULU/Protocols/Protocol_%s/Train_2.txt' % Protocol
    val_list = 'D:/Data/OULU/Protocols/Protocol_%s/Dev_2.txt' % Protocol
    test_list =  'D:/Data/OULU/Protocols/Protocol_%s/Test_2.txt' % Protocol

    train_csv = r'H:/zsw/Data/OULU/CSV/train_%s.csv' % Protocol # the train split file
    val_csv = r'H:/zsw/Data/OULU/CSV/val_%s.csv' % Protocol     # the validation split file
    test_csv = r'H:/zsw/Data/OULU/CSV/test_%s.csv' % Protocol

    train_map_csv = r'H:/zsw/Data/OULU/CSV/train_map_%s.csv' % Protocol # the train split file
    val_map_csv = r'H:/zsw/Data/OULU/CSV/val_map_%s.csv' % Protocol # the validation split file
    test_map_csv = r'H:/zsw/Data/OULU/CSV/test_map_%s.csv' % Protocol

    def oulu_base_process(image_dir, map_dir, list, image_csv, map_csv):
        set = pd.read_csv(list, delimiter=',', header=None)
        map_csv_a = open(map_csv, 'a', encoding='utf-8', newline='')
        map_csv_writer = csv.writer(map_csv_a)
        with open(image_csv, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for i in range(len(set)):
                video_name = str(set.iloc[i, 1])
                labels = int(set.iloc[i, 0])
                if labels == 1:
                    labels = 1
                else:
                    labels = 0

                faces_name = os.listdir(os.path.join(image_dir, video_name))
                for face_name in faces_name:
                    if face_name.split('.')[-1] == 'dat':
                        face_name = face_name.split('.')[0] + '.jpg'
                        map_name = face_name.split('.')[0] + '_depth1D.jpg'
                        map_path = os.path.join(os.path.join(map_dir, video_name), map_name)
                        face_path = os.path.join(os.path.join(image_dir, video_name), face_name)
                        csv_writer.writerow([face_path, labels])
                        map_csv_writer.writerow([map_path, labels])
                    else:
                        continue
        map_csv_a.close()
        return 0

    oulu_base_process(image_dir=train_image_dir, map_dir=train_map_dir, list=train_list,
                 image_csv=train_csv, map_csv=train_map_csv)
    oulu_base_process(image_dir=val_image_dir, map_dir=val_map_dir, list=val_list,
                 image_csv=val_csv, map_csv=val_map_csv)
    oulu_base_process(image_dir=test_image_dir, map_dir=test_map_dir, list=test_list,
                 image_csv=test_csv, map_csv=test_map_csv)


def CASIA_FASD_process():
    train_map_dir = "H:/zsw/Data/CASIA_FASD/CASIA_FASD_DepthMap/train_release/"
    test_map_dir = "H:/zsw/Data/CASIA_FASD/CASIA_FASD_DepthMap/test_release/"

    train_image_dir = "H:/zsw/Data/CASIA_FASD/CASIA_FASD_CropFace256/1.6/train_release/"
    test_image_dir = "H:/zsw/Data/CASIA_FASD/CASIA_FASD_CropFace256/1.6/test_release/"

    train_csv = "H:/zsw/Data/CASIA_FASD/CSV/train.csv"
    test_csv = "H:/zsw/Data/CASIA_FASD/CSV/test.csv"

    train_map_csv = "H:/zsw/Data/CASIA_FASD/CSV/train_map.csv"
    test_map_csv = "H:/zsw/Data/CASIA_FASD/CSV/test_map.csv"

    base_process(image_dir=train_image_dir, map_dir=train_map_dir,
                 image_csv=train_csv, map_csv=train_map_csv)
    base_process(image_dir=test_image_dir, map_dir=test_map_dir,
                 image_csv=test_csv, map_csv=test_map_csv)

def MSU_MFSD_process():
    train_map_dir = "H:/zsw/Data/MSU_MFSD/MSU_MFSD_DepthMap/train/"
    test_map_dir = "H:/zsw/Data/MSU_MFSD/MSU_MFSD_DepthMap/test/"

    train_image_dir = "H:/zsw/Data/MSU_MFSD/MSU_MFSD_CropFace256/1.6/train/"
    test_image_dir = "H:/zsw/Data/MSU_MFSD/MSU_MFSD_CropFace256/1.6/test/"

    train_csv = "H:/zsw/Data/MSU_MFSD/CSV/train.csv"
    test_csv = "H:/zsw/Data/MSU_MFSD/CSV/test.csv"

    train_map_csv = "H:/zsw/Data/MSU_MFSD/CSV/train_map.csv"
    test_map_csv = "H:/zsw/Data/MSU_MFSD/CSV/test_map.csv"

    base_process(image_dir=train_image_dir, map_dir=train_map_dir,
                 image_csv=train_csv, map_csv=train_map_csv)
    base_process(image_dir=test_image_dir, map_dir=test_map_dir,
                 image_csv=test_csv, map_csv=test_map_csv)

def RE_process():
    train_map_dir = "H:/zsw/Data/RE/RE_DepthMap/train/"
    devel_map_dir = "H:/zsw/Data/RE/RE_DepthMap/devel/"
    test_map_dir = "H:/zsw/Data/RE/RE_DepthMap/test/"

    train_image_dir = "H:/zsw/Data/RE/RE_CropFace256/1.6/train/"
    devel_image_dir = "H:/zsw/Data/RE/RE_CropFace256/1.6/devel/"
    test_image_dir = "H:/zsw/Data/RE/RE_CropFace256/1.6/test/"

    train_csv = "H:/zsw/Data/RE/CSV/train.csv"
    test_csv = "H:/zsw/Data/RE/CSV/test.csv"

    train_map_csv = "H:/zsw/Data/RE/CSV/train_map.csv"
    test_map_csv = "H:/zsw/Data/RE/CSV/test_map.csv"

    base_process(image_dir=train_image_dir, map_dir=train_map_dir,
                 image_csv=train_csv, map_csv=train_map_csv)
    base_process(image_dir=devel_image_dir, map_dir=devel_map_dir,
                 image_csv=train_csv, map_csv=train_map_csv)
    base_process(image_dir=test_image_dir, map_dir=test_map_dir,
                 image_csv=test_csv, map_csv=test_map_csv)

if __name__ == '__main__':
    # Modify the following directories to yourselves
    # Oulu_process()
    CASIA_FASD_process()
    # MSU_MFSD_process()
    # RE_process()
