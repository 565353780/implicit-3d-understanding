#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import trimesh
import torch
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from Method.ldif.loss import LDIFLoss
from Method.sdf import reconstruction
from Method.resnet import resnet18_full, model_urls

def roll_pitch_yaw_to_rotation_matrices(roll_pitch_yaw):
    cosines = torch.cos(roll_pitch_yaw)
    sines = torch.sin(roll_pitch_yaw)
    cx, cy, cz = cosines.unbind(-1)
    sx, sy, sz = sines.unbind(-1)
    rotation = torch.stack(
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
         sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
         -sy, cy * sx, cy * cx], -1
    )
    rotation = torch.reshape(rotation, [rotation.shape[0], -1, 3, 3])
    return rotation

def sample_quadric_surface(quadric, center, samples):
    # Sample the quadric surfaces and the RBFs in world space, and composite them.
    samples = samples - center.unsqueeze(2)
    homogeneous_sample_coords = F.pad(samples, [0, 1], "constant", 1)
    half_distance = torch.matmul(quadric, homogeneous_sample_coords.transpose(-1, -2))
    half_distance = half_distance.transpose(-1, -2)
    algebraic_distance = torch.sum(homogeneous_sample_coords * half_distance, -1, keepdim=True)
    return algebraic_distance

def decode_covariance_roll_pitch_yaw(radius, invert=False):
    # Converts 6-D radus vectors to the corresponding covariance matrices.
    d = 1.0 / (radius[..., :3] + 1e-8) if invert else radius[..., :3]
    diag = torch.diag_embed(d)
    rotation = roll_pitch_yaw_to_rotation_matrices(radius[..., 3:6])
    return torch.matmul(torch.matmul(rotation, diag), rotation.transpose(-1, -2))

def sample_cov_bf(center, radius, samples):
    # Samples gaussian radial basis functions at specified coordinates.
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
    # compute shape element influences
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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class StructuredImplicit(object):
    def __init__(self, config, constant, center, radius, iparam, net=None):
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
        sym_elements = elements[:, :self.sym_element_count, ...]
        elements = torch.cat([elements, sym_elements], 1)
        return elements

    def _generate_symgroup_samples(self, samples):
        samples = samples.unsqueeze(1).expand(-1, self.element_count, -1, -1)
        sym_samples = samples[:, :self.sym_element_count].clone()
        sym_samples *= torch.tensor([-1, 1, 1], dtype=torch.float32, device=self.device)
        effective_samples = torch.cat([samples, sym_samples], 1)
        return effective_samples

    def compute_world2local(self):
        tx = torch.eye(3, device=self.device).expand(self.batch_size, self.element_count, -1, -1)
        centers = self.centers.unsqueeze(-1)
        tx = torch.cat([tx, -centers], -1)
        lower_row = torch.tensor([0., 0., 0., 1.], device=self.device).expand(self.batch_size, self.element_count, 1, -1)
        tx = torch.cat([tx, lower_row], -2)

        rotation = roll_pitch_yaw_to_rotation_matrices(self.radii[..., 3:6]).inverse()
        diag = 1.0 / (torch.sqrt(self.radii[..., :3] + 1e-8) + 1e-8)
        scale = torch.diag_embed(diag)

        tx3x3 = torch.matmul(scale, rotation)
        return torch.matmul(homogenize(tx3x3), tx)

    def implicit_values(self, local_samples):
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
        effective_constants = self._tile_for_symgroups(self.constants)
        effective_centers = self._tile_for_symgroups(self.centers)
        effective_radii = self._tile_for_symgroups(self.radii)

        effective_samples = self._generate_symgroup_samples(samples)
        constants_quadrics = torch.zeros(self.batch_size, self.effective_element_count, 4, 4, device=self.device)
        constants_quadrics[:, :, -1:, -1] = effective_constants

        per_element_constants, per_element_weights = compute_shape_element_influences(
            constants_quadrics, effective_centers, effective_radii, effective_samples
        )

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
            batch_mean = sample_embeddings.mean().detach()
            batch_variance = sample_embeddings.var().detach()
            self.running_mean = 0.995 * self.running_mean + 0.005 * batch_mean
            self.running_var = 0.995 * self.running_var + 0.005 * batch_variance
        sample_embeddings = (sample_embeddings - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        return gamma.unsqueeze(1) * sample_embeddings + beta.unsqueeze(1)

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

    def forward(self, embedding, samples):
        sample_embeddings = self.fc1(samples)
        sample_embeddings = self.resnet(embedding, sample_embeddings)
        sample_embeddings = self.bn(embedding, sample_embeddings)
        vals = self.fc2(sample_embeddings)
        return vals

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

