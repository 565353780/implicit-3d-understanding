#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn


from models.registers import MODULES, LOSSES

from Config.ldif_config import TEMP

from Method.base_model import BaseNetwork

class LDIF(BaseNetwork):

    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        optim_spec = self.load_optim_spec(cfg.config, TEMP)
        self.mesh_reconstruction = MODULES.get('LDIF')(cfg, optim_spec)
        self.mesh_reconstruction = nn.DataParallel(self.mesh_reconstruction)

        self.mesh_reconstruction_loss = LOSSES.get('LDIFLoss', 'Null')(1, cfg.config)

        '''freeze submodules or not'''
        self.freeze_modules(cfg)
        return

    def forward(self, data):
        if 'uniform_samples' in data.keys():
            samples = torch.cat([data['near_surface_samples'], data['uniform_samples']], 1)
            len_near_surface = data['near_surface_class'].shape[1]
            est_data = self.mesh_reconstruction(data['img'], data['cls'], samples=samples)
            est_data['near_surface_class'] = est_data['global_decisions'][:, :len_near_surface, ...]
            est_data['uniform_class'] = est_data['global_decisions'][:, len_near_surface:, ...]
        else:
            est_data = self.mesh_reconstruction(data['img'], data['cls'], occnet2gaps=data.get('occnet2gaps'))

        return est_data

    def loss(self, est_data, gt_data):
        '''
        calculate loss of est_out given gt_out.
        '''
        loss = self.mesh_reconstruction_loss(est_data, gt_data)
        total_loss = sum(loss.values())
        for key, item in loss.items():
            loss[key] = item.item()
        return {'total':total_loss, **loss}

