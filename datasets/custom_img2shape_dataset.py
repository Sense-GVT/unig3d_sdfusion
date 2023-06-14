"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import os.path
import json
import csv

import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset

# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class Custom_Img2ShapeDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all', res=64):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res
        self.phase = phase

        dataroot = opt.dataroot

        # TODO: check intersection of train/test csv

        with open(opt.data_info) as f:
            self.info = json.load(f)

        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}

        self.model_list = []
        self.cats_list = []
        self.img_list = []
        self.vox_list = []
        if cat == 'all':
            valid_cats = self.info['cats'].keys()
        else:
            valid_cats = [cat]
        for c in valid_cats:
            synset = self.info['cats'][c]
            model_list_s = []
            render_img_list = []

            with open(f"{self.info['lst_dir']}/{c}_filelists/{synset}_{phase}.lst") as f:
                models_id = f.readlines()
                models_id = [model.strip() for model in models_id]
            model_dir = f"{self.info['raw_dirs']['sdf_dir']}/{c}/SDF/resolution_{self.res}/{synset}"
            for root, subdir, files in os.walk(model_dir):
                for file in files:
                    path = os.path.join(root,file)
                    img_subpath = path[len(model_dir)+1:].replace('.h5','.png')
                    render_img = f"{self.info['raw_dirs']['img_dir']}/{synset}/{img_subpath}"
                    if os.path.exists(path) and os.path.exists(render_img) and (path.split('/')[-2] in models_id or path.split('/')[-1].split('.')[0] in models_id):
                        model_list_s.append(path)
                        render_img_list.append(render_img)
            self.model_list += model_list_s
            self.cats_list += [synset] * len(model_list_s)
            self.img_list += render_img_list

            # print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))
        self.model_list = self.model_list[:self.max_dataset_size]
        # self.text_list = self.text_list[:self.max_dataset_size]
        self.img_list = self.img_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize = transforms.Resize((256, 256))

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if phase == 'train':
            self.transforms_color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            self.transforms = transforms.Compose([
                transforms.RandomAffine(0, scale=(0.7, 1.25), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean, std),
                transforms.Resize((256, 256)),
            ])

        self.transforms_bg = transforms.Compose([
            transforms.RandomCrop(256, pad_if_needed=True, padding_mode='padding_mode'),
            transforms.Normalize(mean, std),
        ])

    def process_img(self, img):
        img_t = self.to_tensor(img)

        _, oh, ow = img_t.shape

        ls = max(oh, ow)

        pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
        pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2

        img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)[0]

        if self.phase == 'train':
            img_fg_mask = (img_t != 0.).float()
            jitter color first
            img_t = self.transforms_color(img_t)
            img_t_with_mask = torch.cat([img_t, img_fg_mask], dim=0)
            img_t_with_mask = self.transforms(img_t_with_mask)
            img_t, img_fg_mask = img_t_with_mask[:3], img_t_with_mask[3:]
            img_fg_mask = self.resize(img_fg_mask)
            img_t = self.normalize(img_t)
            img_t = self.resize(img_t)
        else:
            img_t = self.transforms(img_t)

        return img_t

    def __getitem__(self, index):

        sdf_h5_file = self.model_list[index]
        h5_f = h5py.File(sdf_h5_file, 'r')
        img_path = self.img_list[index]
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        imgs = []
        img_paths = []
        im = Image.open(img_path).convert('RGB')
        im = self.process_img(im)
        imgs.append(im)
        img_paths.append(img_path)
        # allow replacement. cause in test time, we might only see images from one view

        imgs = torch.stack(imgs)
        # img: for one view
        img = imgs[0]
        img_path = img_paths[0]

        ret = {
            'sdf': sdf,
            'img': img,

            'sdf_path': sdf_h5_file,
            'img_path': img_path,
            'img_paths': img_paths,

        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ShapeNetImg2ShapeDataset'
