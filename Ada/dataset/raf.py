import cv2
import numpy as np
from PIL import Image
import os
import copy
import csv

import torchvision
import torch
from .randaugment import RandAugment


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
        self.strong_transfrom = copy.deepcopy(transform)
        self.strong_transfrom_new = copy.deepcopy(transform)
        self.strong_transfrom.transforms.insert(0, RandAugment(3, 5))
        self.strong_transfrom_new.transforms.insert(0, RandAugment(4, 5))

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        out_extra = self.strong_transfrom_new(inp)
        out3 = self.strong_transfrom(inp)
        return out1, out2, out_extra, out3


def get_raf(train_root, train_file_list, test_root, test_file_list, n_labeled, transform_train=None,
            transform_val=None):
    train_labeled_idxs, train_unlabeled_idxs = data_split(train_file_list, int(n_labeled))

    train_labeled_dataset = Dataset_RAF_labeled(train_root, train_file_list, train_labeled_idxs,
                                                transform=transform_train)
    train_unlabeled_dataset = Dataset_RAF_unlabeled(train_root, train_file_list, train_unlabeled_idxs,
                                                    transform=TransformTwice(transform_train))

    test_dataset = Dataset_RAF(test_root, test_file_list, transform=transform_val)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_baseline_raf(train_root, train_file_list, test_root, test_file_list, n_labeled, transform_train=None,
                        transform_val=None):
    train_labeled_idxs, _ = data_split(train_file_list, int(n_labeled))
    train_dataset = Dataset_RAF_labeled(train_root, train_file_list, train_labeled_idxs, transform=transform_train)
    print(f"#Labeled: {len(train_labeled_idxs)}")
    test_dataset = Dataset_RAF(test_root, test_file_list, transform=transform_val)
    return train_dataset, test_dataset


def target_read(path):
    """
    Read old_label from path 获取标签列表
    """
    label_list = []
    with open(path) as f:
        img_label_list = f.read().splitlines()
    for info in img_label_list:
        _, label_name = info.split(' ')
        label_list.append(int(label_name))
    return label_list


# def data_split(filename, n_labeled):
#     """
#     split data into labeled and unlabeled 数据集划分
#     """
#     labels = target_read(filename)
#     labels = np.array(labels)
#     train_labeled_idxs = []
#     train_unlabeled_idxs = []
#
#     for i in range(7):
#         num = 250
#         if i != 1:
#             idxs = np.where(labels == i)[0]
#             np.random.shuffle(idxs)
#             train_labeled_idxs.extend(idxs[:int((n_labeled - num) / 6)])
#             train_unlabeled_idxs.extend(idxs[int((n_labeled - num) / 6):])
#         else:
#             idxs = np.where(labels == i)[0]
#             np.random.shuffle(idxs)
#             train_labeled_idxs.extend(idxs[:num])
#             train_unlabeled_idxs.extend(idxs[num:])
#     np.random.shuffle(train_labeled_idxs)
#     np.random.shuffle(train_unlabeled_idxs)
#
#     return train_labeled_idxs, train_unlabeled_idxs


def data_split(filename, n_labeled):
    """
    split data into labeled and unlabeled 数据集划分
    数据平衡，有标签数据与无标签数据的比例为1:1
    """
    labels = target_read(filename)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(7):
        num = int(n_labeled / 10)
        if i != 1:
            idxs = np.where(labels == i)[0]
            np.random.shuffle(idxs)
            e = int((n_labeled - num) / 6)
            train_labeled_idxs.extend(idxs[:e])

            if idxs.shape[0] > 2*e:
                train_unlabeled_idxs.extend(idxs[e:2*e])
            else:
                train_unlabeled_idxs.extend(idxs[int((n_labeled - num) / 6):])

        else:
            idxs = np.where(labels == i)[0]
            np.random.shuffle(idxs)
            train_labeled_idxs.extend(idxs[:num])
            if idxs.shape[0] > 2*num:
                train_unlabeled_idxs.extend(idxs[num:2*num])
            else:
                train_unlabeled_idxs.extend(idxs[num:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


def img_loader(path):
    """
    Load image from path 加载图片
    """
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = Image.fromarray(img)
            return img
    except IOError:
        print('Cannot load image ' + path)


class Dataset_RAF(torch.utils.data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []

        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_list)


class Dataset_RAF_labeled(Dataset_RAF):
    def __init__(self, root, file_list, indexs, transform=None):
        super(Dataset_RAF_labeled, self).__init__(root, file_list, transform=transform)

        if indexs is not None:
            self.image_list = np.array(self.image_list)[indexs]
            self.label_list = np.array(self.label_list)[indexs]

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class Dataset_RAF_unlabeled(Dataset_RAF_labeled):
    def __init__(self, root, file_list, indexs, transform=None):
        super(Dataset_RAF_unlabeled, self).__init__(root, file_list, indexs, transform=transform)
        self.label_list = np.array([-1 for i in range(len(self.label_list))])
