import os
import csv
import json
import random
import pandas as pd

if __name__ == '__main__':
    # Modify the following directories to yourselves
    train_image_dir = 'H:/zsw/Data/OULU/Image/Train_files/'
    val_image_dir = 'H:/zsw/Data/OULU/Image/Dev_files/'
    test_image_dir = 'H:/zsw/Data/OULU/Image/Test_files/'

    train_map_dir = 'H:/zsw/Data/OULU/Depth_Map/Train_files/'
    val_map_dir = 'H:/zsw/Data/OULU/Depth_Map/Dev_files/'
    test_map_dir = 'H:/zsw/Data/OULU/Depth_Map/Test_files/'

    train_list = 'D:/Data/OULU/Protocols/Protocol_2/Train.txt'
    val_list = 'D:/Data/OULU/Protocols/Protocol_2/Dev.txt'
    test_list =  'D:/Data/OULU/Protocols/Protocol_2/Test.txt'

    train_csv = r'H:/zsw/Data/OULU/CSV/train_2.csv'  # the train split file
    val_csv = r'H:/zsw/Data/OULU/CSV/val_2.csv'      # the validation split file
    test_csv = r'H:/zsw/Data/OULU/CSV/test_2.csv'

    train_map_csv = r'H:/zsw/Data/OULU/CSV/train_map_2.csv'  # the train split file
    val_map_csv = r'H:/zsw/Data/OULU/CSV/val_map_2.csv'  # the validation split file
    test_map_csv = r'H:/zsw/Data/OULU/CSV/test_map_2.csv'



    train_set = pd.read_csv(train_list, delimiter=',', header=None)
    with open(train_csv, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(len(train_set)):
            count = 0
            video_name = str(train_set.iloc[i, 1])
            labels = int(train_set.iloc[i, 0])
            if labels == 1:
                labels = 1
            else:
                labels = 0

            faces_name = os.listdir(os.path.join(train_image_dir, video_name))
            for face_name in faces_name:
                if face_name.split('.')[-1] == 'dat':
                    continue
                face_path = os.path.join(os.path.join(train_image_dir, video_name), face_name)
                csv_writer.writerow([face_path, labels])

    with open(train_map_csv, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(len(train_set)):
            count = 0
            video_name = str(train_set.iloc[i, 1])
            labels = int(train_set.iloc[i, 0])
            if labels == 1:
                labels = 1
            else:
                labels = 0

            maps_name = os.listdir(os.path.join(train_map_dir, video_name))
            for map_name in maps_name:
                map_path = os.path.join(os.path.join(train_map_dir, video_name), map_name)
                csv_writer.writerow([map_path, labels])

    val_set = pd.read_csv(val_list, delimiter=',', header=None)
    with open(val_csv, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(len(val_set)):
            count = 0
            video_name = str(val_set.iloc[i, 1])
            labels = int(val_set.iloc[i, 0])
            if labels == 1:
                labels = 1
            else:
                labels = 0

            faces_name = os.listdir(os.path.join(val_image_dir, video_name))
            for face_name in faces_name:
                if face_name.split('.')[-1] == 'dat':
                    continue
                face_path = os.path.join(os.path.join(val_image_dir, video_name), face_name)
                csv_writer.writerow([face_path, labels])

    with open(val_map_csv, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(len(val_set)):
            count = 0
            video_name = str(val_set.iloc[i, 1])
            labels = int(val_set.iloc[i, 0])
            if labels == 1:
                labels = 1
            else:
                labels = 0

            maps_name = os.listdir(os.path.join(val_map_dir, video_name))
            for map_name in maps_name:
                map_path = os.path.join(os.path.join(val_map_dir, video_name), map_name)
                csv_writer.writerow([map_path, labels])

    test_set = pd.read_csv(test_list, delimiter=',', header=None)
    with open(test_csv, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(len(test_set)):
            count = 0
            video_name = str(test_set.iloc[i, 1])
            labels = int(test_set.iloc[i, 0])
            if labels == 1:
                labels = 1
            else:
                labels = 0

            faces_name = os.listdir(os.path.join(test_image_dir, video_name))
            for face_name in faces_name:
                if face_name.split('.')[-1] == 'dat':
                    continue
                face_path = os.path.join(os.path.join(test_image_dir, video_name), face_name)
                csv_writer.writerow([face_path, labels])

    with open(test_map_csv, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(len(test_set)):
            count = 0
            video_name = str(test_set.iloc[i, 1])
            labels = int(test_set.iloc[i, 0])
            if labels == 1:
                labels = 1
            else:
                labels = 0

            maps_name = os.listdir(os.path.join(test_map_dir, video_name))
            for map_name in maps_name:
                map_path = os.path.join(os.path.join(test_map_dir, video_name), map_name)
                csv_writer.writerow([map_path, labels])
