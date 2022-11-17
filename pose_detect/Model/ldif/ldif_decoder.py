#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import shutil
import tempfile
import subprocess
import numpy as np
import torch.nn as nn

from external.ldif.util import file_util
from external.ldif.inference import extract_mesh

from pose_detect.Method.sdf import reconstruction
from pose_detect.Model.structured_implicit import StructuredImplicit
from pose_detect.Model.occ_net_decoder import OccNetDecoder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LDIFDecoder(nn.Module):

    def __init__(self, config):
        super(LDIFDecoder, self).__init__()
        gauss_kernel_num = 10

        self.config = config

        self.element_count = self.config['model']['element_count']
        self.sym_element_count = self.config['model']['sym_element_count']
        self.effective_element_count = self.element_count + self.sym_element_count
        self.config['model'][
            'effective_element_count'] = self.effective_element_count
        self.implicit_parameter_length = self.config['model'][
            'implicit_parameter_length']
        self.element_embedding_length = 10 + self.implicit_parameter_length
        self.config['model'][
            'analytic_code_len'] = gauss_kernel_num * self.element_count
        self.config['model']['structured_implicit_vector_len'] = \
            self.element_embedding_length * self.element_count

        self.decoder = OccNetDecoder(f_dim=self.implicit_parameter_length)

        self.apply(weights_init)

        self._temp_folder = None
        return

    def eval_implicit_parameters(self, implicit_parameters, samples):
        batch_size, element_count, element_embedding_length = list(
            implicit_parameters.shape)
        sample_count = samples.shape[-2]
        batched_parameters = torch.reshape(
            implicit_parameters,
            [batch_size * element_count, element_embedding_length])
        batched_samples = torch.reshape(
            samples, [batch_size * element_count, sample_count, -1])
        batched_vals = self.decoder(batched_parameters, batched_samples)
        vals = torch.reshape(batched_vals,
                             [batch_size, element_count, sample_count, 1])
        return vals

    def extract_mesh(self,
                     structured_implicit,
                     resolution=64,
                     extent=0.75,
                     num_samples=10000):
        cuda = True
        if cuda:
            mesh = []
            for s in structured_implicit.unbind():
                if self._temp_folder is None:
                    self._temp_folder = tempfile.mktemp(dir='/dev/shm')
                    os.makedirs(self._temp_folder)
                    self.decoder.write_occnet_file(
                        os.path.join(self._temp_folder, 'serialized.occnet'))
                    shutil.copy('./external/ldif/ldif2mesh/ldif2mesh',
                                self._temp_folder)
                si_path = os.path.join(self._temp_folder, 'ldif.txt')
                grd_path = os.path.join(self._temp_folder, 'grid.grd')

                s.savetxt(si_path)
                cmd = (
                    f"{os.path.join(self._temp_folder, 'ldif2mesh')} {si_path}"
                    f" {os.path.join(self._temp_folder, 'serialized.occnet')}"
                    f' {grd_path} -resolution {resolution} -extent {extent}')
                subprocess.check_output(cmd, shell=True)
                _, volume = file_util.read_grd(grd_path)
                _, m = extract_mesh.marching_cubes(volume, extent)
                mesh.append(m)
        else:
            mesh = reconstruction(structured_implicit, resolution,
                                  np.array([-extent] * 3),
                                  np.array([extent] * 3), num_samples, False)
        return mesh

    def forward(self, structured_implicit_activations, samples=None):
        return_dict = {}

        structured_implicit = StructuredImplicit.from_activation(
            self.config, structured_implicit_activations, self)
        return_dict['structured_implicit'] = structured_implicit.dict()

        if samples is not None:
            global_decisions, local_outputs = structured_implicit.class_at_samples(
                samples, True)
            return_dict.update({
                'global_decisions': global_decisions,
                'element_centers': structured_implicit.centers
            })
            return return_dict

        resolution = self.config['data'].get('marching_cube_resolution', 128)
        mesh = self.extract_mesh(structured_implicit, resolution,
                                 self.config['data']['bounding_box'])

        mesh_coordinates_results = []
        faces = []
        for m in mesh:
            mesh_coordinates_results.append(
                torch.from_numpy(m.vertices).type(torch.float32).transpose(
                    -1, -2).to(structured_implicit.device))
            faces.append(
                torch.from_numpy(m.faces).to(structured_implicit.device) + 1)

        return_dict.update({
            'mesh':
            mesh,
            'mesh_coordinates_results': [
                mesh_coordinates_results,
            ],
            'faces':
            faces,
            'element_centers':
            structured_implicit.centers
        })

        return_dict.update({
            'sdf': mesh[0],
            'mat': mesh[1],
            'element_centers': structured_implicit.centers
        })
        return return_dict
