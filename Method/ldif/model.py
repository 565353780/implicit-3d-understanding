#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import struct
import tempfile
import shutil
import subprocess
import trimesh
import torch
import numpy as np
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from configs.data_config import pix3d_n_classes
from models.modules import resnet
from models.modules.resnet import model_urls
from net_utils.misc import weights_init
from external.ldif.representation.structured_implicit_function import StructuredImplicit
from external.ldif.inference import extract_mesh
from external.PIFu.lib import mesh_util
from external.ldif.util import file_util, camera_util

from Method.ldif.loss import LDIFLoss

def sample_quadric_surface(quadric, center, samples):
    # Sample the quadric surfaces and the RBFs in world space, and composite them.
    # (ldif.representation.quadrics.sample_quadric_surface)
    samples = samples - center.unsqueeze(2)
    homogeneous_sample_coords = F.pad(samples, [0, 1], "constant", 1)
    half_distance = torch.matmul(quadric, homogeneous_sample_coords.transpose(-1, -2))
    half_distance = half_distance.transpose(-1, -2)
    algebraic_distance = torch.sum(homogeneous_sample_coords * half_distance, -1, keepdim=True)
    return algebraic_distance

def decode_covariance_roll_pitch_yaw(radius, invert=False):
    # Converts 6-D radus vectors to the corresponding covariance matrices.
    # (ldif.representation.quadrics.decode_covariance_roll_pitch_yaw
    d = 1.0 / (radius[..., :3] + 1e-8) if invert else radius[..., :3]
    diag = torch.diag_embed(d)
    rotation = camera_util.roll_pitch_yaw_to_rotation_matrices(radius[..., 3:6])
    return torch.matmul(torch.matmul(rotation, diag), rotation.transpose(-1, -2))

def sample_cov_bf(center, radius, samples):
    # Samples gaussian radial basis functions at specified coordinates.
    # (ldif.representation.quadrics.sample_cov_bf)
    diff = samples - center.unsqueeze(2)
    x, y, z = diff.unbind(-1)

    inv_cov = decode_covariance_roll_pitch_yaw(radius, invert=True)
    inv_cov = torch.reshape(inv_cov, [inv_cov.shape[0], -1, 1, 9])
    c00, c01, c02, _, c11, c12, _, _, c22 = inv_cov.unbind(-1)
    dist = (x * (c00 * x + c01 * y + c02 * z)
            + y * (c01 * x + c11 * y + c12 * z)
            + z * (c02 * x + c12 * y + c22 * z))
    dist = torch.exp(-0.5 * dist)
    return dist.unsqueeze(-1)

def compute_shape_element_influences(quadrics, centers, radii, samples):
    # compute shape element influences (ldif.representation.quadrics.compute_shape_element_influences)
    sampled_quadrics = sample_quadric_surface(quadrics, centers, samples)
    sampled_rbfs = sample_cov_bf(centers, radii, samples)
    return sampled_quadrics, sampled_rbfs

def homogenize(m):
  # Adds homogeneous coordinates to a [..., N,N] matrix.
  m = F.pad(m, [0, 1, 0, 1], "constant", 0)
  m[..., -1, -1] = 1
  return m

def _unflatten(config, vector):
    return torch.split(vector, [1, 3, 6, config['model']['implicit_parameter_length']], -1)

class StructuredImplicit(object):
    def __init__(self, config, constant, center, radius, iparam, net=None):
        # (ldif.representation.structured_implicit_function.StructuredImplicit.from_activation)
        self.config = config
        self.implicit_parameter_length = config['model']['implicit_parameter_length']
        self.element_count = config['model']['element_count']
        self.sym_element_count = config['model']['sym_element_count']

        self.constants = constant
        self.radii = radius
        self.centers = center
        self.iparams = iparam
        self.effective_element_count = self.element_count + self.sym_element_count
        self.device = constant.device
        self.batch_size = constant.size(0)
        self.net = net
        self._packed_vector = None
        self._analytic_code = None
        self._all_centers = None

    @classmethod
    def from_packed_vector(cls, config, packed_vector, net):
        """Parse an already packed vector (NOT a network activation)."""
        constant, center, radius, iparam = _unflatten(config, packed_vector)
        return cls(config, constant, center, radius, iparam, net)

    @classmethod
    def from_activation(cls, config, activation, net):
        constant, center, radius, iparam = _unflatten(config, activation)
        constant = -torch.abs(constant)
        radius_var = torch.sigmoid(radius[..., :3])
        radius_var = 0.15 * radius_var
        radius_var = radius_var * radius_var
        max_euler_angle = np.pi / 4.0
        radius_rot = torch.clamp(radius[..., 3:], -max_euler_angle, max_euler_angle)
        radius = torch.cat([radius_var, radius_rot], -1)
        new_center = center / 2
        return cls(config, constant, new_center, radius, iparam, net)

    def _tile_for_symgroups(self, elements):
        # Tiles an input tensor along its element dimension based on symmetry
        # (ldif.representation.structured_implicit_function._tile_for_symgroups)
        sym_elements = elements[:, :self.sym_element_count, ...]
        elements = torch.cat([elements, sym_elements], 1)
        return elements

    def _generate_symgroup_samples(self, samples):
        samples = samples.unsqueeze(1).expand(-1, self.element_count, -1, -1)
        sym_samples = samples[:, :self.sym_element_count].clone()
        # sym_samples *= torch.tensor([1, 1, -1], dtype=torch.float32, device=self.device)  # reflect across the XY plane
        sym_samples *= torch.tensor([-1, 1, 1], dtype=torch.float32, device=self.device)  # reflect across the YZ plane
        effective_samples = torch.cat([samples, sym_samples], 1)
        return effective_samples

    def compute_world2local(self):
        tx = torch.eye(3, device=self.device).expand(self.batch_size, self.element_count, -1, -1)
        centers = self.centers.unsqueeze(-1)
        tx = torch.cat([tx, -centers], -1)
        lower_row = torch.tensor([0., 0., 0., 1.], device=self.device).expand(self.batch_size, self.element_count, 1, -1)
        tx = torch.cat([tx, lower_row], -2)

        # Compute a rotation transformation
        rotation = camera_util.roll_pitch_yaw_to_rotation_matrices(self.radii[..., 3:6]).inverse()
        diag = 1.0 / (torch.sqrt(self.radii[..., :3] + 1e-8) + 1e-8)
        scale = torch.diag_embed(diag)

        # Apply both transformations and return the transformed points.
        tx3x3 = torch.matmul(scale, rotation)
        return torch.matmul(homogenize(tx3x3), tx)

    def implicit_values(self, local_samples):
        # Computes the implicit values given local input locations.
        iparams = self._tile_for_symgroups(self.iparams)
        values = self.net.eval_implicit_parameters(iparams, local_samples)
        return values

    @property
    def all_centers(self):
        if self._all_centers is None:
            sym_centers = self.centers[:, :self.sym_element_count].clone()
            sym_centers[:, :, 0] *= -1  # reflect across the YZ plane
            self._all_centers = torch.cat([self.centers, sym_centers], 1)
        return self._all_centers

    def class_at_samples(self, samples, apply_class_transfer=True):
        # (ldif.representation.structured_implicit_function.StructuredImplicit.class_at_samples)
        effective_constants = self._tile_for_symgroups(self.constants)
        effective_centers = self._tile_for_symgroups(self.centers)
        effective_radii = self._tile_for_symgroups(self.radii)

        effective_samples = self._generate_symgroup_samples(samples)
        constants_quadrics = torch.zeros(self.batch_size, self.effective_element_count, 4, 4, device=self.device)
        constants_quadrics[:, :, -1:, -1] = effective_constants

        per_element_constants, per_element_weights = compute_shape_element_influences(
            constants_quadrics, effective_centers, effective_radii, effective_samples
        )

        # We currently have constants, weights with shape:
        # [batch_size, element_count, sample_count, 1].
        # We need to use the net to get a same-size grid of offsets.
        # The input samples to the net must have shape
        # [batch_size, element_count, sample_count, 3], while the current samples
        # have shape [batch_size, sample_count, 3]. This is because each sample
        # should be evaluated in the relative coordinate system of the
        # The world2local transformations for each element. Shape [B, EC, 4, 4].
        effective_world2local = self._tile_for_symgroups(self.compute_world2local())
        local_samples = torch.matmul(F.pad(effective_samples, [0, 1], "constant", 1),
                                     effective_world2local.transpose(-1, -2))[..., :3]
        implicit_values = self.implicit_values(local_samples)

        residuals = 1 + implicit_values
        local_decisions = per_element_constants * per_element_weights * residuals
        local_weights = per_element_weights
        sdf = torch.sum(local_decisions, 1)
        if apply_class_transfer:
            sdf = torch.sigmoid(100 * (sdf + 0.07))

        return sdf, (local_decisions, local_weights)

    @property
    def vector(self):
        if self._packed_vector is None:
            self._packed_vector = torch.cat([self.constants, self.centers, self.radii, self.iparams], -1)
        return self._packed_vector

    @property
    def analytic_code(self):
        if self._analytic_code is None:
            self._analytic_code = torch.cat([self.constants, self.centers, self.radii], -1)
        return self._analytic_code

    def savetxt(self, path):
        assert self.vector.shape[0] == 1
        sif_vector = self.vector.squeeze().cpu().numpy()
        sif_vector[:, 4:7] = np.sqrt(np.maximum(sif_vector[:, 4:7], 0))
        out = 'SIF\n%i %i %i\n' % (self.element_count, 0, self.implicit_parameter_length)
        for row_idx in range(self.element_count):
            row = ' '.join(10 * ['%.9g']) % tuple(sif_vector[row_idx, :10].tolist())
            symmetry = int(row_idx < self.sym_element_count)
            row += ' %i' % symmetry
            implicit_params = ' '.join(self.implicit_parameter_length * ['%.9g']) % (
                tuple(sif_vector[row_idx, 10:].tolist()))
            row += ' ' + implicit_params
            row += '\n'
            out += row
        file_util.writetxt(path, out)

    def unbind(self):
        return [StructuredImplicit.from_packed_vector(self.config, self.vector[i:i+1], self.net)
                for i in range(self.vector.size(0))]

    def __getitem__(self, item):
        return StructuredImplicit.from_packed_vector(self.config, self.vector[item], self.net)

    def dict(self):
        return {'constant': self.constants, 'radius': self.radii, 'center': self.centers, 'iparam': self.iparams}

class BatchedCBNLayer(nn.Module):
    def __init__(self, f_dim=32):
        super(BatchedCBNLayer, self).__init__()
        self.fc_beta = nn.Linear(f_dim, f_dim)
        self.fc_gamma = nn.Linear(f_dim, f_dim)
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))

    def forward(self, shape_embedding, sample_embeddings):
        beta = self.fc_beta(shape_embedding)
        gamma = self.fc_gamma(shape_embedding)
        if self.training:
            batch_mean, batch_variance = sample_embeddings.mean().detach(), sample_embeddings.var().detach()
            self.running_mean = 0.995 * self.running_mean + 0.005 * batch_mean
            self.running_var = 0.995 * self.running_var + 0.005 * batch_variance
        sample_embeddings = (sample_embeddings - self.running_mean) / torch.sqrt(self.running_var + 1e-5)

        out = gamma.unsqueeze(1) * sample_embeddings + beta.unsqueeze(1)

        return out

class BatchedOccNetResnetLayer(nn.Module):
    def __init__(self, f_dim=32):
        super(BatchedOccNetResnetLayer, self).__init__()
        self.bn1 = BatchedCBNLayer(f_dim=f_dim)
        self.fc1 = nn.Linear(f_dim, f_dim)
        self.bn2 = BatchedCBNLayer(f_dim=f_dim)
        self.fc2 = nn.Linear(f_dim, f_dim)

    def forward(self, shape_embedding, sample_embeddings):
        sample_embeddings = self.bn1(shape_embedding, sample_embeddings)
        init_sample_embeddings = sample_embeddings

        sample_embeddings = torch.relu(sample_embeddings)
        sample_embeddings = self.fc1(sample_embeddings)
        sample_embeddings = self.bn2(shape_embedding, sample_embeddings)

        sample_embeddings = torch.relu(sample_embeddings)
        sample_embeddings = self.fc2(sample_embeddings)

        return init_sample_embeddings + sample_embeddings

class OccNetDecoder(nn.Module):
    def __init__(self, f_dim=32):
        super(OccNetDecoder, self).__init__()
        self.fc1 = nn.Linear(3, f_dim)
        self.resnet = BatchedOccNetResnetLayer(f_dim=f_dim)
        self.bn = BatchedCBNLayer(f_dim=f_dim)
        self.fc2 = nn.Linear(f_dim, 1)

    def write_occnet_file(self, path):
        """Serializes an occnet network and writes it to disk."""
        f = file_util.open_file(path, 'wb')

        def write_fc_layer(layer):
            weights = layer.weight.t().cpu().numpy()
            biases = layer.bias.cpu().numpy()
            f.write(weights.astype('f').tostring())
            f.write(biases.astype('f').tostring())

        def write_cbn_layer(layer):
            write_fc_layer(layer.fc_beta)
            write_fc_layer(layer.fc_gamma)
            running_mean = layer.running_mean.item()
            running_var = layer.running_var.item()
            f.write(struct.pack('ff', running_mean, running_var))

        # write_header
        f.write(struct.pack('ii', 1, self.fc1.out_features))
        # write_input_layer
        write_fc_layer(self.fc1)
        # write_resnet
        write_cbn_layer(self.resnet.bn1)
        write_fc_layer(self.resnet.fc1)
        write_cbn_layer(self.resnet.bn2)
        write_fc_layer(self.resnet.fc2)
        # write_cbn_layer
        write_cbn_layer(self.bn)
        # write_activation_layer
        weights = self.fc2.weight.t().cpu().numpy()
        bias = self.fc2.bias.data.item()
        f.write(weights.astype('f').tostring())
        f.write(struct.pack('f', bias))
        f.close()

    def forward(self, embedding, samples):
        sample_embeddings = self.fc1(samples)
        sample_embeddings = self.resnet(embedding, sample_embeddings)
        sample_embeddings = self.bn(embedding, sample_embeddings)
        vals = self.fc2(sample_embeddings)
        return vals

class LDIF_SubNet(nn.Module):
    def __init__(self, config, optim_spec=None, n_classes=pix3d_n_classes,
                 pretrained_encoder=True):
        super(LDIF_SubNet, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Module parameters'''
        self.config = config
        self.bottleneck_size = self.config['model'].get('bottleneck_size', 2048)
        self.config['model']['bottleneck_size'] = self.bottleneck_size
        self.element_count = self.config['model']['element_count']
        self.sym_element_count = self.config['model']['sym_element_count']
        self.effective_element_count = self.element_count + self.sym_element_count
        self.config['model']['effective_element_count'] = self.effective_element_count
        self.implicit_parameter_length = self.config['model']['implicit_parameter_length']
        self.element_embedding_length = 10 + self.implicit_parameter_length
        self.config['model']['analytic_code_len'] = 10 * self.element_count
        self.config['model']['structured_implicit_vector_len'] = \
            self.element_embedding_length * self.element_count
        self._temp_folder = None

        '''Modules'''
        self.encoder = resnet.resnet18_full(
            pretrained=False, num_classes=self.bottleneck_size,
            input_channels=4 if self.config['data'].get('mask', False) else 3)
        self.mlp = nn.Sequential(
            nn.Linear(self.bottleneck_size + n_classes, self.bottleneck_size), nn.LeakyReLU(0.2, True),
            nn.Linear(self.bottleneck_size, self.bottleneck_size), nn.LeakyReLU(0.2, True),
            nn.Linear(self.bottleneck_size, self.element_count * self.element_embedding_length)
        )
        self.decoder = OccNetDecoder(f_dim=self.implicit_parameter_length)

        # initialize weight
        self.apply(weights_init)

        # initialize resnet
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

    def eval_implicit_parameters(self, implicit_parameters, samples):
        batch_size, element_count, element_embedding_length = list(implicit_parameters.shape)
        sample_count = samples.shape[-2]
        batched_parameters = torch.reshape(implicit_parameters, [batch_size * element_count, element_embedding_length])
        batched_samples = torch.reshape(samples, [batch_size * element_count, sample_count, -1])
        batched_vals = self.decoder(batched_parameters, batched_samples)
        vals = torch.reshape(batched_vals, [batch_size, element_count, sample_count, 1])
        return vals

    def extract_mesh(self, structured_implicit, resolution=64, extent=0.75, num_samples=10000,
                     cuda=True, marching_cube=True):
        if cuda:
            mesh = []
            for s in structured_implicit.unbind():
                if self._temp_folder is None:
                    self._temp_folder = tempfile.mktemp(dir='/dev/shm')
                    os.makedirs(self._temp_folder)
                    self.decoder.write_occnet_file(os.path.join(self._temp_folder, 'serialized.occnet'))
                    shutil.copy('./external/ldif/ldif2mesh/ldif2mesh', self._temp_folder)
                si_path = os.path.join(self._temp_folder, 'ldif.txt')
                grd_path = os.path.join(self._temp_folder, 'grid.grd')

                s.savetxt(si_path)
                cmd = (f"{os.path.join(self._temp_folder, 'ldif2mesh')} {si_path}" 
                       f" {os.path.join(self._temp_folder, 'serialized.occnet')}"
                       f' {grd_path} -resolution {resolution} -extent {extent}')
                subprocess.check_output(cmd, shell=True)
                _, volume = file_util.read_grd(grd_path)
                _, m = extract_mesh.marching_cubes(volume, extent)
                mesh.append(m)
        else:
            mesh = mesh_util.reconstruction(structured_implicit=structured_implicit, resolution=resolution,
                                            b_min=np.array([-extent] * 3), b_max=np.array([extent] * 3),
                                            use_octree=True, num_samples=num_samples, marching_cube=marching_cube)
        return mesh

    def forward(self, image=None, size_cls=None, samples=None, occnet2gaps=None, structured_implicit=None,
                resolution=None, cuda=True, reconstruction='mesh', apply_class_transfer=True):
        return_dict = {}
        # predict structured_implicit
        return_structured_implicit = structured_implicit
        if isinstance(structured_implicit, dict):
            structured_implicit = StructuredImplicit(config=self.config, **structured_implicit, net=self)
        elif structured_implicit is None or isinstance(structured_implicit, bool):
            # encoder (ldif.model.model.StructuredImplicitModel.forward)
            # image encoding (ldif.nets.cnn.early_fusion_cnn)
            embedding = self.encoder(image)
            return_dict['ldif_afeature'] = embedding
            embedding = torch.cat([embedding, size_cls], 1)
            structured_implicit_activations = self.mlp(embedding)
            structured_implicit_activations = torch.reshape(
                structured_implicit_activations, [-1, self.element_count, self.element_embedding_length])
            return_dict['structured_implicit_activations'] = structured_implicit_activations

            # SIF decoder
            structured_implicit = StructuredImplicit.from_activation(
                self.config, structured_implicit_activations, self)
        else:
            raise NotImplementedError

        return_dict['structured_implicit'] = structured_implicit.dict()

        # if only want structured_implicit
        if return_structured_implicit is True:
            return return_dict

        # predict class or mesh
        if samples is not None:
            global_decisions, local_outputs = structured_implicit.class_at_samples(samples, apply_class_transfer)
            return_dict.update({'global_decisions': global_decisions,
                                'element_centers': structured_implicit.centers})
            return return_dict
        elif reconstruction is not None:
            if resolution is None:
                resolution =  self.config['data'].get('marching_cube_resolution', 128)
            mesh = self.extract_mesh(structured_implicit, extent=self.config['data']['bounding_box'],
                                     resolution=resolution, cuda=cuda, marching_cube=reconstruction == 'mesh')
            if reconstruction == 'mesh':
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
            elif reconstruction == 'sdf':
                return_dict.update({'sdf': mesh[0], 'mat': mesh[1], 'element_centers': structured_implicit.centers})
            else:
                raise NotImplementedError
            return return_dict
        else:
            return return_dict

    def __del__(self):
        if self._temp_folder is not None:
            shutil.rmtree(self._temp_folder)

class LDIF(nn.Module):

    def __init__(self, config, mode):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(LDIF, self).__init__()
        self.config = config
        self.mode = mode

        optim_spec = self.load_optim_spec()
        self.mesh_reconstruction = LDIF_SubNet(config, optim_spec)
        self.mesh_reconstruction = nn.DataParallel(self.mesh_reconstruction)

        self.mesh_reconstruction_loss = LDIFLoss(1, self.config)

        '''freeze submodules or not'''
        self.freeze_modules()
        return

    def freeze_modules(self):
        '''
        Freeze modules in training
        '''
        if self.mode == 'train':
            freeze_layers = self.config['train']['freeze']
            for layer in freeze_layers:
                if not hasattr(self, layer):
                    continue
                for param in getattr(self, layer).parameters():
                    param.requires_grad = False
                print('The module: %s is fixed.' % (layer))
        return True

    def set_mode(self):
        '''
        Set train/eval mode for the network.
        :param phase: train or eval
        :return:
        '''
        freeze_layers = self.config['train']['freeze']
        for name, child in self.named_children():
            if name in freeze_layers:
                child.train(False)

        # turn off BatchNorm if batch_size == 1.
        if self.config[self.mode]['batch_size'] == 1:
            for m in self.modules():
                if m._get_name().find('BatchNorm') != -1:
                    m.eval()
        return True

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

        print(str(set([key.split('.')[0] for key in model_dict if key not in pretrained_dict])) + ' subnet missed.')
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_optim_spec(self):
        if self.mode != 'train':
            return None

        return self.config['optimizer']

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

