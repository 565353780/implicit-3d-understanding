#!/usr/bin/env python
# -*- coding: utf-8 -*-

from net_utils.libs import get_bdb_form_from_corners, recover_points_to_world_sys


class ReconLoss(object):

    def __init__(self, weight=1, config=None):
        self.weight = weight
        self.config = config
        return

    def __call__(self, est_data, gt_data, extra_results):
        if gt_data['mask_flag'] == 0:
            point_loss = 0.
        else:
            # get the world coordinates for each 3d object.
            bdb3D_form = get_bdb_form_from_corners(
                extra_results['bdb3D_result'], gt_data['mask_status'])
            obj_points_in_world_sys = recover_points_to_world_sys(
                bdb3D_form, est_data['meshes'])
            point_loss = 100 * get_point_loss(
                obj_points_in_world_sys, extra_results['cam_R_result'],
                gt_data['K'], gt_data['depth_maps'], bdb3D_form,
                gt_data['split'], gt_data['obj_masks'], gt_data['mask_status'])

            # remove samples without depth map
            if torch.isnan(point_loss):
                point_loss = 0.

        return {'mesh_loss': point_loss}
