#!/usr/bin/env python
# -*- coding: utf-8 -*-

DEVICE = {
    'device': 'cuda',
}

DATA = {
    'num_workers': 8,
    'batch_size': 24,
}

MODEL = {
    #  'model_save_path': './out/ldif/22062721592286/',
    'bottleneck_size': 1536,
    'element_count': 32,
    'sym_element_count': 16,
    'implicit_parameter_length': 32,
    'uniform_loss_weight': 1.0,
    'near_surface_loss_weight': 0.1,
    'lowres_grid_inside_loss_weight': 0.2,
    'inside_box_loss_weight': 10.0,
}

OPTIMIZER = {
    'method': 'Adam',
    'lr': 2e-4,
    'betas': [0.9, 0.999],
    'eps': 1e-08,
    'weight_decay': 0.0,
}

SCHEDULER = {
    'patience': 50,
    'factor': 0.5,
    'threshold': 0.002,
}

TRAIN = {
    'epochs': 40000,
    'phase': 'all',
    'freeze': [],
    'batch_size': 24,
}

VAL = {
    'phase': 'all',
    'batch_size': 24,
}

TEST = {
    'phase': 'all',
    'batch_size': 1,
}

LOG = {
    'vis_path': 'visualization',
    'save_results': True,
    'vis_step': 100,
    'print_step': 50,
    'path': './out/ldif/',
    'save_checkpoint': True,
}

LDIF_CONFIG = {
    'device': DEVICE,
    'data': DATA,
    'model': MODEL,
    'optimizer': OPTIMIZER,
    'scheduler': SCHEDULER,
    'train': TRAIN,
    'val': VAL,
    'test': TEST,
    'log': LOG,
}

TEMP = {
    'method': 'LDIF',
    'loss': 'LDIFLoss',
    'bottleneck_size': 1536,
    'element_count': 32,
    'sym_element_count': 16,
    'implicit_parameter_length': 32,
    'uniform_loss_weight': 1.0,
    'near_surface_loss_weight': 0.1,
    'lowres_grid_inside_loss_weight': 0.2,
    'inside_box_loss_weight': 10.0,
}

