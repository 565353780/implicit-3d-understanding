#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LDIFLoss(object):

    def __init__(self, config=None, weight=1):
        self.config = config
        self.weight = weight
        return

    def getLoss(self, est_data, gt_data):
        uniform_sample_loss = nn.MSELoss()(est_data['uniform_class'],
                                           gt_data['uniform_class'])
        uniform_sample_loss *= self.config['model']['uniform_loss_weight']

        near_surface_sample_loss = nn.MSELoss()(est_data['near_surface_class'],
                                                gt_data['near_surface_class'])
        near_surface_sample_loss *= self.config['model'][
            'near_surface_loss_weight']

        element_centers = est_data['element_centers']

        #  dim = [batch_size, sample_count, 4]
        xyzw_samples = F.pad(element_centers, [0, 1], "constant", 1)

        # dim = [batch_size, sample_count, 3]
        xyzw_samples = torch.matmul(xyzw_samples,
                                    gt_data['world2grid'])[..., :3]

        grid = gt_data['grid']
        scale_fac = torch.Tensor(list(grid.shape)[1:]).to(
            element_centers.device) / 2 - 0.5
        xyzw_samples /= scale_fac

        # dim = [batch_size, 1, 1, sample_count, 3]
        xyzw_samples = xyzw_samples.unsqueeze(1).unsqueeze(1)

        grid = grid.unsqueeze(1)

        gt_sdf_at_centers = F.grid_sample(grid,
                                          xyzw_samples,
                                          mode='bilinear',
                                          padding_mode='zeros')

        gt_sdf_at_centers = torch.where(
            gt_sdf_at_centers >
            self.config['data']['coarse_grid_spacing'] / 1.1,
            gt_sdf_at_centers,
            torch.zeros(1).to(gt_sdf_at_centers.device))
        gt_sdf_at_centers *= self.config['model'][
            'lowres_grid_inside_loss_weight']

        element_center_lowres_grid_inside_loss = torch.mean(
            (gt_sdf_at_centers + 1e-04)**2) + 1e-05

        bounding_box = self.config['data']['bounding_box']
        lower, upper = -bounding_box, bounding_box
        lower_error = torch.max(lower - element_centers, torch.zeros(1).cuda())
        upper_error = torch.max(element_centers - upper, torch.zeros(1).cuda())
        bounding_box_constraint_error = lower_error * lower_error + upper_error * upper_error
        bounding_box_error = torch.mean(bounding_box_constraint_error)
        inside_box_loss = self.config['model'][
            'inside_box_loss_weight'] * bounding_box_error

        return {
            'uniform_sample_loss': uniform_sample_loss,
            'near_surface_sample_loss': near_surface_sample_loss,
            'fixed_bounding_box_loss': inside_box_loss,
            'lowres_grid_inside_loss': element_center_lowres_grid_inside_loss
        }

    def __call__(self, est_data, gt_data):
        loss = self.getLoss(est_data, gt_data)

        if 'cad_est_data' not in est_data.keys():
            return loss

        cad_loss = self.getLoss(est_data['cad_est_data'], gt_data)

        loss['cad_uniform_sample_loss'] = cad_loss['uniform_sample_loss']
        loss['cad_near_surface_sample_loss'] = cad_loss[
            'near_surface_sample_loss']
        loss['cad_fixed_bounding_box_loss'] = cad_loss[
            'fixed_bounding_box_loss']
        loss['cad_lowres_grid_inside_loss'] = cad_loss[
            'lowres_grid_inside_loss']
        return loss


class LDIFReconLoss(object):

    def __init__(self, weight=1):
        self.weight = weight
        return

    def __call__(self, est_data, gt_data, extra_results):
        ldif = est_data['mgn']
        device = gt_data['image'].device

        loss_type = 'classmse'
        scale_before_func = 100.0
        phy_loss_samples = 128
        phy_loss_objects = 4
        surface_optimize = True
        sdf_data = {}

        bdb3D_form = get_bdb_form_from_corners(extra_results['bdb3D_result'])
        structured_implicit = est_data['structured_implicit']
        ldif_center, ldif_coef = est_data['obj_center'], est_data['obj_coef']
        if 'ldif_sampling_points' in est_data:
            inside_samples = est_data['ldif_sampling_points']
        else:
            obj_center = ldif_center.clone()
            obj_center[:, 2] *= -1
            inside_samples = get_phy_loss_samples(
                ldif,
                structured_implicit,
                obj_center,
                ldif_coef,
                phy_loss_samples,
                surface_optimize=surface_optimize)

        # put points to other objects' coor
        inside_samples[:, :, 2] *= -1
        obj_samples = recover_points_to_world_sys(bdb3D_form, inside_samples,
                                                  ldif_center, ldif_coef)
        max_sample_points = (gt_data['split'][:, 1] - gt_data['split'][:, 0] -
                             1).max() * obj_samples.shape[1]
        if max_sample_points == 0:
            sdf_data['ldif_phy_loss'] = None
        else:
            est_sdf = []
            for start, end in gt_data['split']:
                assert end > start
                if end > start + 1:
                    centroids = bdb3D_form['centroid'][start:end]
                    centroids = centroids.unsqueeze(0).expand(
                        len(centroids), -1, -1)
                    distances = F.pairwise_distance(
                        centroids.reshape(-1, 3),
                        centroids.transpose(0, 1).reshape(-1, 3),
                        2).reshape(len(centroids), len(centroids))
                    for obj_ind in range(start, end):
                        other_obj_dis = distances[obj_ind - start]
                        _, nearest = torch.sort(other_obj_dis)
                        other_obj_sample = obj_samples[start:end].index_select(
                            0, nearest[1:phy_loss_objects + 1]).reshape(-1, 3)
                        other_obj_sample = recover_points_to_obj_sys(
                            {
                                k: v[obj_ind:obj_ind + 1]
                                for k, v in bdb3D_form.items()
                            }, other_obj_sample.unsqueeze(0),
                            ldif_center[obj_ind:obj_ind + 1],
                            ldif_coef[obj_ind:obj_ind + 1])
                        other_obj_sample[:, :, 2] *= -1
                        sdf = ldif(
                            samples=other_obj_sample,
                            structured_implicit=structured_implicit[
                                obj_ind:obj_ind + 1].dict(),
                            apply_class_transfer=False,
                        )['global_decisions']
                        est_sdf.append(sdf.squeeze())
            if len(est_sdf) == 0:
                sdf_data['ldif_phy_loss'] = None
            else:
                est_sdf = torch.cat(est_sdf) + 0.07
                est_sdf[est_sdf > 0] = 0
                gt_sdf = torch.full(est_sdf.shape,
                                    0.,
                                    device=device,
                                    dtype=torch.float32)
                sdf_data['ldif_phy_loss'] = (est_sdf, gt_sdf)

        # compute final loss
        loss = {}
        if not isinstance(loss_type, list):
            loss_type = [loss_type] * len(sdf_data)
        for lt, (k, sdf) in zip(loss_type, sdf_data.items()):
            if sdf is None:
                loss[k] = 0.
            else:
                est_sdf, gt_sdf = sdf

                if 'class' in lt:
                    est_sdf = torch.sigmoid(scale_before_func * est_sdf) - 0.5
                    gt_sdf[gt_sdf > 0] = 0.5
                    gt_sdf[gt_sdf < 0] = -0.5
                elif 'sdf' in lt:
                    est_sdf = scale_before_func * est_sdf
                else:
                    raise NotImplementedError

                if 'mse' in lt:
                    point_loss = nn.MSELoss()(est_sdf, gt_sdf)
                elif 'l1' in lt:
                    point_loss = nn.L1Loss()(est_sdf, gt_sdf)
                elif 'sl1' in lt:
                    point_loss = nn.SmoothL1Loss()(est_sdf, gt_sdf)
                else:
                    raise NotImplementedError

                loss[k] = point_loss

        return loss
