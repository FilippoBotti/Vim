# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data import Dataset
from PIL import Image
from itertools import chain
from pathlib import Path

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class CelebaDataset(Dataset):
    def __init__(self, root, mask_dir, img_size, transform=None):
        self.images, self.masks = self._make_dataset(root, mask_dir)
        self.transform = transform
        self.img_size = img_size

    def _make_dataset(self, root, root_mask_dir):
        fnames, masks = [], []
        cls_fnames = listdir(root)
        mask_fnames = listdir(root_mask_dir)

        cls_fnames.sort()
        mask_fnames.sort()
        fnames += cls_fnames
        masks += mask_fnames
        return list(fnames), list(masks)

    def __getitem__(self, index):
        fname = self.images[index]
        mask_fname = self.masks[index]
        img = Image.open(fname).convert('RGB')
        mask = Image.open(mask_fname).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask

    def __len__(self):
        return len(self.images)
    
class Edges2Shoes(Dataset):
    def __init__(self, root, img_size, transform=None):
        self.images = self._make_dataset(root)
        self.transform = transform
        self.img_size = img_size

    def _make_dataset(self, dir):
        images = []
        for fname in os.listdir(dir):
            path = os.path.join(dir, fname)
            images.append(path)
        return images

    def __getitem__(self, index):
        # read a image given a random integer index
        coupled_images = self.images[index]
        
        AB = Image.open(coupled_images).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        edge = self.transform(A)
        shoes = self.transform(B)

        return edge, shoes

    def __len__(self):
        return len(self.images)
    
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
    if args.data_set in ['EDGES2SHOES', 'CelebA-HQ']:
        transform = build_transform(False, args)
    else:
        transform = build_transform(is_train, args)
        
    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
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
    elif args.data_set == 'EDGES2SHOES':
        root = os.path.join(args.data_path, 'train' if is_train else ('val' if args.eval else 'val'))
        dataset = Edges2Shoes(root, img_size=256, transform=transform)
        nb_classes = 0
    elif args.data_set == 'CelebA-HQ':
        root = os.path.join(args.data_path, 'train' if is_train else ('val' if args.eval else 'val'))
        images_root = os.path.join(root, "A")
        masks_root = os.path.join(root, "B")
        dataset = CelebaDataset(images_root, masks_root, img_size=256, transform=transform)
        nb_classes = 0
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.data_set in ['EGDES2SHOES', 'CelebA-HQ']:
            size = args.input_size
        else:
            size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
