# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import pandas as pd
import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

import numpy as np 
from PIL import Image


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'SEA':
        dataset = SEADataset(args.data_path, transform=transform)

#    return dataset, nb_classes
    return dataset


class SEADataset(Dataset):
    def __init__(self, path = None, transform = None):
        super().__init__()
        #, datetime_2015.15.pt, datetime_2015.28.pt, datetime_2015.41.pt, datetime_2015.02.pt, datetime_2015.16.pt, datetime_2015.29.pt, datetime_2015.42.pt, datetime_2015.03.pt, datetime_2015.17.pt, datetime_2015.2.pt, datetime_2015.43.pt, datetime_2015.04.pt, datetime_2015.18.pt, datetime_2015.31.pt, datetime_2015.44.pt, datetime_2015.05.pt, datetime_2015.19.pt, datetime_2015.32.pt, datetime_2015.45.pt, datetime_2015.06.pt, datetime_2015.1.pt, datetime_2015.33.pt, datetime_2015.46.pt, datetime_2015.07.pt, datetime_2015.21.pt, datetime_2015.34.pt, datetime_2015.47.pt, datetime_2015.08.pt, datetime_2015.22.pt, datetime_2015.35.pt, datetime_2015.48.pt, datetime_2015.09.pt, datetime_2015.23.pt, datetime_2015.36.pt, datetime_2015.49.pt, datetime_2015.11.pt, datetime_2015.24.pt, datetime_2015.37.pt, datetime_2015.4.pt, datetime_2015.12.pt, datetime_2015.25.pt, datetime_2015.38.pt, datetime_2015.51.pt, datetime_2015.13.pt, datetime_2015.26.pt, datetime_2015.39.pt, datetime_2015.52.pt, datetime_2015.14.pt, datetime_2015.27.pt, datetime_2015.3.pt, datetime_2015.5.pt 
        #self.list_files = ['datetime_2015.01.pt', 'datetime_2015.15.pt', 'datetime_2015.28.pt', 'datetime_2015.41.pt', 'datetime_2015.02.pt', 'datetime_2015.16.pt', 'datetime_2015.29.pt', 'datetime_2015.42.pt', 'datetime_2015.03.pt', 'datetime_2015.17.pt', 'datetime_2015.2.pt', 'datetime_2015.43.pt', 'datetime_2015.04.pt', 'datetime_2015.18.pt', 'datetime_2015.31.pt', 'datetime_2015.44.pt', 'datetime_2015.05.pt', 'datetime_2015.19.pt', 'datetime_2015.32.pt', 'datetime_2015.45.pt', 'datetime_2015.06.pt', 'datetime_2015.1.pt', 'datetime_2015.33.pt', 'datetime_2015.46.pt', 'datetime_2015.07.pt', 'datetime_2015.21.pt', 'datetime_2015.34.pt', 'datetime_2015.47.pt', 'datetime_2015.08.pt', 'datetime_2015.22.pt', 'datetime_2015.35.pt', 'datetime_2015.48.pt', 'datetime_2015.09.pt', 'datetime_2015.23.pt', 'datetime_2015.36.pt', 'datetime_2015.49.pt', 'datetime_2015.11.pt', 'datetime_2015.24.pt', 'datetime_2015.37.pt', 'datetime_2015.4.pt', 'datetime_2015.12.pt', 'datetime_2015.25.pt', 'datetime_2015.38.pt', 'datetime_2015.51.pt', 'datetime_2015.13.pt', 'datetime_2015.26.pt', 'datetime_2015.39.pt', 'datetime_2015.52.pt', 'datetime_2015.14.pt', 'datetime_2015.27.pt', 'datetime_2015.3.pt', 'datetime_2015.5.pt'] 
        self.list_files = ['datetime_2015.01.pt', 'datetime_2015.15.pt', 'datetime_2015.28.pt', 'datetime_2015.41.pt', 'datetime_2015.02.pt', 'datetime_2015.16.pt', 'datetime_2015.29.pt', 'datetime_2015.42.pt', 'datetime_2015.03.pt', 'datetime_2015.17.pt', 'datetime_2015.2.pt', 'datetime_2015.43.pt', 'datetime_2015.04.pt', 'datetime_2015.18.pt', 'datetime_2015.31.pt', 'datetime_2015.44.pt', 'datetime_2015.05.pt', 'datetime_2015.19.pt', 'datetime_2015.32.pt', 'datetime_2015.45.pt', 'datetime_2015.06.pt', 'datetime_2015.1.pt', 'datetime_2015.33.pt', 'datetime_2015.46.pt', 'datetime_2015.07.pt', 'datetime_2015.21.pt', 'datetime_2015.34.pt', 'datetime_2015.47.pt', 'datetime_2015.08.pt', 'datetime_2015.22.pt', 'datetime_2015.35.pt', 'datetime_2015.09.pt', 'datetime_2015.23.pt', 'datetime_2015.36.pt', 'datetime_2015.11.pt', 'datetime_2015.24.pt', 'datetime_2015.37.pt', 'datetime_2015.4.pt', 'datetime_2015.12.pt', 'datetime_2015.25.pt', 'datetime_2015.38.pt', 'datetime_2015.13.pt', 'datetime_2015.26.pt', 'datetime_2015.39.pt', 'datetime_2015.14.pt', 'datetime_2015.27.pt', 'datetime_2015.3.pt'] 
        self.transform = transform


        if path is not None:
            self.path = path
        else:
            raise Exception("Paths should be given as input to initialize the SEA class.")

    def __len__(self):
        # so that len(dataset) returns the size of the dataset
        return len(self.list_files) # SEE

    def __getitem__(self, index):
        # to support the indexing such that dataset[i] can be used to get ith sample

        # select sample
        ID = self.list_files[index]

        # load data and get label
        data = torch.load(self.path + ID)


        N, C, D, H, W = data.shape

        ds = data.reshape(C, D, H, W)
        ds[:, 30, :, :] = ds[:, 29, :, :]

        X = ds[:3, 0, :30, :30]
        y = ds[0, 0, :30, :30]

        y = torch.unsqueeze(y, 0)
        #D, H, W = y.shape
        #y = y.reshape(1, D, H, W)      

        return X, y



def build_transform(is_train, args):
    #resize_im = args.input_size > 32
    t = []
    if is_train:
        t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        # this should always dispatch to transforms_imagenet_train
    #    transform = create_transform(
    #        input_size=args.input_size,
    #        is_training=True,
    #        color_jitter=args.color_jitter,
    #        auto_augment=args.aa,
    #        interpolation=args.train_interpolation,
    #        re_prob=args.reprob,
    #        re_mode=args.remode,
    #        re_count=args.recount,
    #    )
    #    if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
    #        transform.transforms[0] = transforms.RandomCrop(
    #            args.input_size, padding=4)
    #    return transform

    #if resize_im:
    #    size = int((256 / 224) * args.input_size)
    #    t.append(
    #        transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
    #    )
    #    t.append(transforms.CenterCrop(args.input_size))

    #t.append(transforms.ToTensor())
    return transforms.Compose(t)
