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

