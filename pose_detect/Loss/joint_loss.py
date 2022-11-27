#!/usr/bin/env python
# -*- coding: utf-8 -*-


from net_utils.libs import \
    get_rotation_matix_result, get_bdb_3d_result, \
    get_bdb_2d_result, physical_violation


class JointLoss(object):

    def __init__(self, weight=1):
        self.weight = weight
        return

    def __call__(self, est_data, gt_data, bins_tensor, layout_results):
        # predicted camera rotation
        cam_R_result = get_rotation_matix_result(bins_tensor,
                                                 gt_data['pitch_cls'],
                                                 est_data['pitch_reg_result'],
                                                 gt_data['roll_cls'],
                                                 est_data['roll_reg_result'])

        # projected center
        P_result = torch.stack(
            ((gt_data['bdb2D_pos'][:, 0] + gt_data['bdb2D_pos'][:, 2]) / 2 -
             (gt_data['bdb2D_pos'][:, 2] - gt_data['bdb2D_pos'][:, 0]) *
             est_data['offset_2D_result'][:, 0],
             (gt_data['bdb2D_pos'][:, 1] + gt_data['bdb2D_pos'][:, 3]) / 2 -
             (gt_data['bdb2D_pos'][:, 3] - gt_data['bdb2D_pos'][:, 1]) *
             est_data['offset_2D_result'][:, 1]), 1)

        # retrieved 3D bounding box
        bdb3D_result, _ = get_bdb_3d_result(
            bins_tensor, gt_data['ori_cls'], est_data['ori_reg_result'],
            gt_data['centroid_cls'], est_data['centroid_reg_result'],
            gt_data['size_cls'], est_data['size_reg_result'], P_result,
            gt_data['K'], cam_R_result, gt_data['split'])

        # 3D bounding box corner loss
        corner_loss = 5 * cls_reg_ratio * reg_criterion(
            bdb3D_result, gt_data['bdb3D'])

        # 2D bdb loss
        bdb2D_result = get_bdb_2d_result(bdb3D_result, cam_R_result,
                                         gt_data['K'], gt_data['split'])
        bdb2D_loss = 20 * cls_reg_ratio * reg_criterion(
            bdb2D_result, gt_data['bdb2D_from_3D_gt'])

        # physical violation loss
        phy_violation, phy_gt = physical_violation(
            layout_results['lo_bdb3D_result'], bdb3D_result, gt_data['split'])
        phy_loss = 20 * mse_criterion(phy_violation, phy_gt)

        return {'phy_loss':phy_loss, 'bdb2D_loss':bdb2D_loss, 'corner_loss':corner_loss},\
               {'cam_R_result':cam_R_result, 'bdb3D_result':bdb3D_result}
