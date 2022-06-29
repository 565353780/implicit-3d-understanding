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
    'model_save_path': './out/ldif/22062721592286/',
    'bottleneck_size': 1536,
    'element_count': 32,
    'sym_element_count': 16,
    'implicit_parameter_length': 32,
    'uniform_loss_weight': 1.0,
    'near_surface_loss_weight': 0.1,
    'lowres_grid_inside_loss_weight': 0.2,
    'inside_box_loss_weight': 10.0,
}

LOG = {
    'path': './out/ldif/',
}

LDIF_CONFIG = {
    'device': DEVICE,
    'data': DATA,
    'model': MODEL,
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

