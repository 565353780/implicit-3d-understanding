#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.utils.data
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from configs.pix3d_config import Config
from tqdm import tqdm

from configs.data_config import pix3d_n_classes
from external.ldif.util import gaps_util, file_util

class PIX3DLDIF(Dataset):
    def __init__(self, config, mode):
        '''
        initiate PIX3DLDIF dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        self.mode = mode
        if mode == 'val':
            mode = 'test'
        split_file = os.path.join(config['data']['split'], mode + '.json')
        with open(split_file) as file:
            split = json.load(file)

        config_data = Config('pix3d')
        with open(config_data.metadata_file, 'r') as file:
            metadatas = json.load(file)
        ids = [int(os.path.basename(file).split('.')[0]) for file in split if 'flipped' not in file]
        sample_info = []
        skipped = 0
        for id in tqdm(ids):
            metadata = metadatas[id]
            info = {}

            rel_img = metadata['img'].replace('img/', '').split('/')
            model_folder = '.'.join(os.path.splitext(metadata['model'])[0].split('/')[-2:])
            rel_folder = os.path.join(config_data.root_path, 'ldif', *rel_img[:-1], model_folder)
            img_name = os.path.splitext(rel_img[-1])[0]
            info['img_path'] = os.path.join(rel_folder, img_name + '.npy')
            info['nss_points_path'] = os.path.join(rel_folder, 'nss_points.sdf')
            info['uniform_points_path'] = os.path.join(rel_folder, 'uniform_points.sdf')
            info['coarse_grid_path'] = os.path.join(rel_folder, 'coarse_grid.grd')
            info['occnet2gaps_path'] = os.path.join(rel_folder, 'orig_to_gaps.txt')
            watertight = self.config['data'].get('watertight',
                                                 self.config['model']['mesh_reconstruction']['method'] == 'LDIF')
            info['mesh_path'] = os.path.join(rel_folder, 'mesh_orig.ply' if watertight else 'mesh_normalized.ply')
            ext_mgnet = 'mgn' if watertight else 'org'
            info['gt_3dpoints_path'] = os.path.join(rel_folder, f'gt_3dpoints.{ext_mgnet}')
            info['densities_path'] = os.path.join(rel_folder, f'densities.{ext_mgnet}')
            if not all([os.path.exists(path) for path in info.values()]) :
                skipped += 1
                continue

            info['sample_id'] = id
            info['class_id'] = config_data.classnames.index(metadata['category'])
            info['class_name'] = metadata['category']
            sample_info.append(info)
        print(f'{skipped}/{len(ids)} missing samples')
        self.split = sample_info

    def __len__(self):
        return len(self.split)

class LDIF_Dataset(PIX3DLDIF):
    def __init__(self, config, mode):
        super(LDIF_Dataset, self).__init__(config, mode)
        HEIGHT_PATCH = 256
        WIDTH_PATCH = 256
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode=='train':
            self.data_transforms = transforms.Compose([
                transforms.Resize((280, 280)),
                transforms.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.data_transforms = transforms.Compose([
                transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __getitem__(self, index):
        sample_info = self.split[index]
        sample = {}

        image = np.load(sample_info['img_path'])
        image = Image.fromarray(image)
        sample['img'] = self.data_transforms(image)

        cls_codes = torch.zeros(pix3d_n_classes)
        cls_codes[sample_info['class_id']] = 1
        sample['cls'] = cls_codes

        if self.config['model']['mesh_reconstruction']['method'] == 'LDIF':
            if self.mode == 'test':
                sample['mesh'] = file_util.read_mesh(sample_info['mesh_path'])
                occnet2gaps = file_util.read_txt_to_np(sample_info['occnet2gaps_path'])
                sample['occnet2gaps'] = np.reshape(occnet2gaps, [4, 4])
            else:
                near_surface_samples = gaps_util.read_pts_file(sample_info['nss_points_path'])
                p_ids = np.random.choice(near_surface_samples.shape[0],
                                         self.config['data']['near_surface_samples'],
                                         replace=False)
                near_surface_samples = near_surface_samples[p_ids, :]
                sample['near_surface_class'] = (near_surface_samples[:, 3:] > 0).astype(np.float32)
                sample['near_surface_samples'] = near_surface_samples[:, :3]

                uniform_samples = gaps_util.read_pts_file(sample_info['uniform_points_path'])
                p_ids = np.random.choice(uniform_samples.shape[0],
                                         self.config['data']['uniform_samples'],
                                         replace=False)
                uniform_samples = uniform_samples[p_ids, :]
                sample['uniform_class'] = (uniform_samples[:, 3:] > 0).astype(np.float32)
                sample['uniform_samples'] = uniform_samples[:, :3]

                sample['world2grid'], sample['grid'] = file_util.read_grd(sample_info['coarse_grid_path'])
                # from external.PIFu.lib import sample_util
                # sample_util.save_samples_truncted_prob('near_surface_samples.ply', sample['near_surface_samples'], sample['near_surface_class'])
                # sample_util.save_samples_truncted_prob('uniform_samples.ply', sample['uniform_samples'], sample['uniform_class'])
        elif self.config['model']['mesh_reconstruction']['method'] == 'DensTMNet':
            sample['sequence_id'] = sample_info['sample_id']
            sample['mesh_points'] = np.fromfile(
                sample_info['gt_3dpoints_path'], dtype=np.float).reshape(-1, 3).astype(np.float32)
            sample['densities'] = np.fromfile(
                sample_info['densities_path'], dtype=np.float).astype(np.float32)
            if self.mode == 'train':
                p_ids = np.random.choice(sample['mesh_points'].shape[0], 5000, replace=False)
                sample['mesh_points'] = sample['mesh_points'][p_ids, :]
                sample['densities'] = sample['densities'][p_ids]
        else:
            raise NotImplementedError
        sample.update(sample_info)
        return sample

