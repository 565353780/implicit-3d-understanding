#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn


from models.registers import MODULES, LOSSES

from Config.ldif_config import TEMP

class LDIF(nn.Module):

    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(LDIF, self).__init__()
        self.cfg = cfg

        optim_spec = self.load_optim_spec(cfg.config, TEMP)
        self.mesh_reconstruction = MODULES.get('LDIF')(cfg, optim_spec)
        self.mesh_reconstruction = nn.DataParallel(self.mesh_reconstruction)

        self.mesh_reconstruction_loss = LOSSES.get('LDIFLoss', 'Null')(1, cfg.config)

        '''freeze submodules or not'''
        self.freeze_modules(cfg)
        return

    def freeze_modules(self, cfg):
        '''
        Freeze modules in training
        '''
        if cfg.config['mode'] == 'train':
            freeze_layers = cfg.config['train']['freeze']
            for layer in freeze_layers:
                if not hasattr(self, layer):
                    continue
                for param in getattr(self, layer).parameters():
                    param.requires_grad = False
                cfg.log_string('The module: %s is fixed.' % (layer))

    def set_mode(self):
        '''
        Set train/eval mode for the network.
        :param phase: train or eval
        :return:
        '''
        freeze_layers = self.cfg.config['train']['freeze']
        for name, child in self.named_children():
            if name in freeze_layers:
                child.train(False)

        # turn off BatchNorm if batch_size == 1.
        if self.cfg.config[self.cfg.config['mode']]['batch_size'] == 1:
            for m in self.modules():
                if m._get_name().find('BatchNorm') != -1:
                    m.eval()

    def load_weight(self, pretrained_model):
        model_dict = self.state_dict()
        pretrained_dict = {}
        for k, v in pretrained_model.items():
            if k not in model_dict:
                print(f"{k} will not be loaded because not exist in target model")
            elif model_dict[k].shape != v.shape:
                print(f"{k} will not be loaded because source shape {model_dict[k].shape} != taget shape {v.shape}")
            else:
                pretrained_dict[k] = v

        self.cfg.log_string(
            str(set([key.split('.')[0] for key in model_dict if key not in pretrained_dict])) + ' subnet missed.')
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_optim_spec(self, config, net_spec):
        # load specific optimizer parameters
        if config['mode'] == 'train':
            if 'optimizer' in net_spec.keys():
                optim_spec = net_spec['optimizer']
            else:
                optim_spec = config['optimizer']  # else load default optimizer
        else:
            optim_spec = None

        return optim_spec

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

