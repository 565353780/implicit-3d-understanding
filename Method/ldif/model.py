#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from Method.image_encoder.model import ImageLDIFEncoder
from Method.ldif_decoder.model import LDIFDecoder
from Method.ldif.loss import LDIFLoss

class ImageLDIF(nn.Module):
    def __init__(self, config, n_classes):
        super(ImageLDIF, self).__init__()

        self.config = config

        self.image_encoder = ImageLDIFEncoder(config, n_classes)
        self.ldif_decoder = LDIFDecoder(config)
        return

    def forward(self, image, size_cls, samples=None):
        return_dict = self.image_encoder.forward(image, size_cls)

        decoder_return_dict = self.ldif_decoder.forward(return_dict['structured_implicit_activations'], samples)

        return_dict.update(decoder_return_dict)
        return return_dict

class LDIF(nn.Module):
    def __init__(self, config, mode):
        super(LDIF, self).__init__()
        self.config = config
        self.mode = mode

        self.mesh_reconstruction = ImageLDIF(config, 9)

        self.mesh_reconstruction_loss = LDIFLoss(1, self.config)

        self.set_mode()
        return

    def set_mode(self):
        if self.config[self.mode]['batch_size'] == 1:
            for m in self.modules():
                if m._get_name().find('BatchNorm') != -1:
                    m.eval()
        return True

    def forward(self, data):
        if 'uniform_samples' not in data.keys():
            return self.mesh_reconstruction(data['img'], data['cls'])

        samples = torch.cat([data['near_surface_samples'], data['uniform_samples']], 1)
        est_data = self.mesh_reconstruction(data['img'], data['cls'], samples=samples)
        len_near_surface = data['near_surface_class'].shape[1]
        est_data['near_surface_class'] = est_data['global_decisions'][:, :len_near_surface, ...]
        est_data['uniform_class'] = est_data['global_decisions'][:, len_near_surface:, ...]
        return est_data

    def loss(self, est_data, gt_data):
        loss = self.mesh_reconstruction_loss(est_data, gt_data)
        total_loss = sum(loss.values())
        for key, item in loss.items():
            loss[key] = item.item()
        return {'total':total_loss, **loss}

