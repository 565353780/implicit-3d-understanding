#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LDIFLoss(object):
    def __init__(self, weight=1, config=None):
        self.weight = weight
        self.config = config
        return

    def __call__(self, est_data, gt_data):
        # calculate loss (ldif.training.loss.compute_loss)
        uniform_sample_loss = nn.MSELoss()(est_data['uniform_class'], gt_data['uniform_class'])
        uniform_sample_loss *= self.config['model']['uniform_loss_weight']

        near_surface_sample_loss = nn.MSELoss()(est_data['near_surface_class'], gt_data['near_surface_class'])
        near_surface_sample_loss *= self.config['model']['near_surface_loss_weight']

        element_centers = est_data['element_centers']
        xyzw_samples = F.pad(element_centers, [0, 1], "constant", 1) # 维度为 [batch_size, sample_count, 4]
        xyzw_samples = torch.matmul(xyzw_samples, gt_data['world2grid'])[..., :3] # 维度为 [batch_size, sample_count, 3]
        # xyzw_samples = torch.matmul(torch.Tensor([[[1.2] * 3 + [1]], ] * xyzw_samples.shape[0]).to(element_centers.device), gt_data['world2grid'])[..., :3] # 测试边界
        grid = gt_data['grid']
        scale_fac = torch.Tensor(list(grid.shape)[1:]).to(element_centers.device) / 2 - 0.5
        xyzw_samples /= scale_fac
        xyzw_samples = xyzw_samples.unsqueeze(1).unsqueeze(1) # 维度为 [batch_size, 1, 1, sample_count, 3]
        grid = grid.unsqueeze(1)
        gt_sdf_at_centers = F.grid_sample(grid, xyzw_samples, mode='bilinear', padding_mode='zeros') # https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.grid_sample
        gt_sdf_at_centers = torch.where(gt_sdf_at_centers > self.config['data']['coarse_grid_spacing'] / 1.1,
                                        gt_sdf_at_centers, torch.zeros(1).to(gt_sdf_at_centers.device))
        gt_sdf_at_centers *= self.config['model']['lowres_grid_inside_loss_weight']
        element_center_lowres_grid_inside_loss = torch.mean((gt_sdf_at_centers + 1e-04) ** 2) + 1e-05

        bounding_box = self.config['data']['bounding_box']
        lower, upper = -bounding_box, bounding_box
        lower_error = torch.max(lower - element_centers, torch.zeros(1).cuda())
        upper_error = torch.max(element_centers - upper, torch.zeros(1).cuda())
        bounding_box_constraint_error = lower_error * lower_error + upper_error * upper_error
        bounding_box_error = torch.mean(bounding_box_constraint_error)
        inside_box_loss = self.config['model']['inside_box_loss_weight'] * bounding_box_error

        return {'uniform_sample_loss': uniform_sample_loss,
                'near_surface_sample_loss': near_surface_sample_loss,
                'fixed_bounding_box_loss': inside_box_loss,
                'lowres_grid_inside_loss':element_center_lowres_grid_inside_loss}

