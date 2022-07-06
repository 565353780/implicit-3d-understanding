#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import trimesh
import torch
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from Method.resnet import resnet18_full, model_urls
from Method.sdf import reconstruction
from Method.sif.model import StructuredImplicit
from Method.occnet.model import OccNetDecoder
from Method.ldif.loss import LDIFLoss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LDIFSDF(nn.Module):
    def __init__(self, config, n_classes=9,
                 pretrained_encoder=True):
        super(LDIFSDF, self).__init__()
        gauss_kernel_num = 10

        self.config = config

        self.bottleneck_size = self.config['model'].get('bottleneck_size', 2048)
        self.config['model']['bottleneck_size'] = self.bottleneck_size
        self.element_count = self.config['model']['element_count']
        self.sym_element_count = self.config['model']['sym_element_count']
        self.effective_element_count = self.element_count + self.sym_element_count
        self.config['model']['effective_element_count'] = self.effective_element_count
        self.implicit_parameter_length = self.config['model']['implicit_parameter_length']
        self.element_embedding_length = 10 + self.implicit_parameter_length
        self.config['model']['analytic_code_len'] = gauss_kernel_num * self.element_count
        self.config['model']['structured_implicit_vector_len'] = \
            self.element_embedding_length * self.element_count

        self._temp_folder = None

        self.encoder = resnet18_full(
            pretrained=False, num_classes=self.bottleneck_size,
            input_channels=4 if self.config['data'].get('mask', False) else 3)
        self.mlp = nn.Sequential(
            nn.Linear(self.bottleneck_size + n_classes, self.bottleneck_size), nn.LeakyReLU(0.2, True),
            nn.Linear(self.bottleneck_size, self.bottleneck_size), nn.LeakyReLU(0.2, True),
            nn.Linear(self.bottleneck_size, self.element_count * self.element_embedding_length)
        )
        self.decoder = OccNetDecoder(f_dim=self.implicit_parameter_length)

        self.apply(weights_init)

        if pretrained_encoder:
            pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
            model_dict = self.encoder.state_dict()
            if pretrained_dict['conv1.weight'].shape != model_dict['conv1.weight'].shape:
                model_dict['conv1.weight'][:,:3,...] = pretrained_dict['conv1.weight']
                pretrained_dict.pop('conv1.weight')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and not k.startswith('fc.')}
            model_dict.update(pretrained_dict)
            self.encoder.load_state_dict(model_dict)
        return

    def eval_implicit_parameters(self, implicit_parameters, samples):
        batch_size, element_count, element_embedding_length = list(implicit_parameters.shape)
        sample_count = samples.shape[-2]
        batched_parameters = torch.reshape(implicit_parameters, [batch_size * element_count, element_embedding_length])
        batched_samples = torch.reshape(samples, [batch_size * element_count, sample_count, -1])
        batched_vals = self.decoder(batched_parameters, batched_samples)
        vals = torch.reshape(batched_vals, [batch_size, element_count, sample_count, 1])
        return vals

    def extract_mesh(self, structured_implicit, resolution=64, extent=0.75, num_samples=10000, marching_cube=True):
        mesh = reconstruction(structured_implicit, resolution,
                              np.array([-extent] * 3), np.array([extent] * 3),
                              num_samples, marching_cube)
        return mesh

    def forward(self, image, size_cls, samples=None):
        return_dict = {}

        embedding = self.encoder(image)
        return_dict['ldif_afeature'] = embedding
        embedding = torch.cat([embedding, size_cls], 1)
        structured_implicit_activations = self.mlp(embedding)
        structured_implicit_activations = torch.reshape(
            structured_implicit_activations, [-1, self.element_count, self.element_embedding_length])
        return_dict['structured_implicit_activations'] = structured_implicit_activations

        structured_implicit = StructuredImplicit.from_activation(
            self.config, structured_implicit_activations, self)

        return_dict['structured_implicit'] = structured_implicit.dict()

        if samples is not None:
            global_decisions, local_outputs = structured_implicit.class_at_samples(samples, True)
            return_dict.update({'global_decisions': global_decisions,
                                'element_centers': structured_implicit.centers})
            return return_dict

        resolution =  self.config['data'].get('marching_cube_resolution', 128)
        mesh = self.extract_mesh(structured_implicit, extent=self.config['data']['bounding_box'],
                                 resolution=resolution, marching_cube=False)

        return_dict.update({'sdf': mesh[0], 'mat': mesh[1],
                            'element_centers': structured_implicit.centers})
        return return_dict

    def __del__(self):
        if self._temp_folder is not None:
            shutil.rmtree(self._temp_folder)
        return

class LDIF_SubNet(LDIFSDF):
    def __init__(self, config, n_classes=9,
                 pretrained_encoder=True):
        super(LDIF_SubNet, self).__init__(config, n_classes, pretrained_encoder)
        return

    def forward(self, image, size_cls, samples=None, occnet2gaps=None):
        return_dict = {}

        embedding = self.encoder(image)
        return_dict['ldif_afeature'] = embedding
        embedding = torch.cat([embedding, size_cls], 1)
        structured_implicit_activations = self.mlp(embedding)
        structured_implicit_activations = torch.reshape(
            structured_implicit_activations, [-1, self.element_count, self.element_embedding_length])
        return_dict['structured_implicit_activations'] = structured_implicit_activations

        structured_implicit = StructuredImplicit.from_activation(
            self.config, structured_implicit_activations, self)

        return_dict['structured_implicit'] = structured_implicit.dict()

        if samples is not None:
            global_decisions, local_outputs = structured_implicit.class_at_samples(samples, True)
            return_dict.update({'global_decisions': global_decisions,
                                'element_centers': structured_implicit.centers})
            return return_dict

        resolution =  self.config['data'].get('marching_cube_resolution', 128)
        mesh = self.extract_mesh(structured_implicit, extent=self.config['data']['bounding_box'],
                                 resolution=resolution, marching_cube=True)

        if occnet2gaps is not None:
            mesh = [m.apply_transform(t.inverse().cpu().numpy()) if not isinstance(m, trimesh.primitives.Sphere) else m
                    for m, t in zip(mesh, occnet2gaps)]

        mesh_coordinates_results = []
        faces = []
        for m in mesh:
            mesh_coordinates_results.append(
                torch.from_numpy(m.vertices).type(torch.float32).transpose(-1, -2).to(structured_implicit.device))
            faces.append(torch.from_numpy(m.faces).to(structured_implicit.device) + 1)
        return_dict.update({'mesh': mesh, 'mesh_coordinates_results': [mesh_coordinates_results, ],
                            'faces': faces, 'element_centers': structured_implicit.centers})
        return return_dict

class LDIF(nn.Module):
    def __init__(self, config, mode):
        super(LDIF, self).__init__()
        self.config = config
        self.mode = mode

        self.mesh_reconstruction = nn.DataParallel(LDIF_SubNet(config))

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
            return self.mesh_reconstruction(data['img'], data['cls'],
                                            occnet2gaps=data.get('occnet2gaps'))

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

