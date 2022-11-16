#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from configs.data_config import cls_reg_ratio
from net_utils.libs import get_layout_bdb_sunrgbd


class PoseLoss(object):

    def __init__(self, config=None, weight=1):
        self.config = config
        self.weight = weight

        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.reg_criterion = nn.SmoothL1Loss(reduction='mean')
        return

    def cls_reg_loss(self, cls_result, cls_gt, reg_result, reg_gt):
        cls_loss = self.cls_criterion(cls_result, cls_gt)
        if len(reg_result.size()) == 3:
            reg_result = torch.gather(
                reg_result, 1,
                cls_gt.view(reg_gt.size(0), 1,
                            1).expand(reg_gt.size(0), 1, reg_gt.size(1)))
        else:
            reg_result = torch.gather(
                reg_result, 1,
                cls_gt.view(reg_gt.size(0), 1).expand(reg_gt.size(0), 1))
        reg_result = reg_result.squeeze(1)
        reg_loss = self.reg_criterion(reg_result, reg_gt)
        return cls_loss, cls_reg_ratio * reg_loss

    def __call__(self, est_data, gt_data, bins_tensor):
        pitch_cls_loss, pitch_reg_loss = self.cls_reg_loss(
            est_data['pitch_cls_result'], gt_data['pitch_cls'],
            est_data['pitch_reg_result'], gt_data['pitch_reg'])
        roll_cls_loss, roll_reg_loss = self.cls_reg_loss(
            est_data['roll_cls_result'], gt_data['roll_cls'],
            est_data['roll_reg_result'], gt_data['roll_reg'])
        lo_ori_cls_loss, lo_ori_reg_loss = self.cls_reg_loss(
            est_data['lo_ori_cls_result'], gt_data['lo_ori_cls'],
            est_data['lo_ori_reg_result'], gt_data['lo_ori_reg'])
        lo_centroid_loss = self.reg_criterion(
            est_data['lo_centroid_result'],
            gt_data['lo_centroid']) * cls_reg_ratio
        lo_coeffs_loss = self.reg_criterion(
            est_data['lo_coeffs_result'], gt_data['lo_coeffs']) * cls_reg_ratio

        lo_bdb3D_result = get_layout_bdb_sunrgbd(
            bins_tensor, est_data['lo_ori_reg_result'], gt_data['lo_ori_cls'],
            est_data['lo_centroid_result'], est_data['lo_coeffs_result'])
        # layout bounding box corner loss
        lo_corner_loss = cls_reg_ratio * self.reg_criterion(
            lo_bdb3D_result, gt_data['lo_bdb3D'])

        return {
            'pitch_cls_loss': pitch_cls_loss,
            'pitch_reg_loss': pitch_reg_loss,
            'roll_cls_loss': roll_cls_loss,
            'roll_reg_loss': roll_reg_loss,
            'lo_ori_cls_loss': lo_ori_cls_loss,
            'lo_ori_reg_loss': lo_ori_reg_loss,
            'lo_centroid_loss': lo_centroid_loss,
            'lo_coeffs_loss': lo_coeffs_loss,
            'lo_corner_loss': lo_corner_loss
        }, {
            'lo_bdb3D_result': lo_bdb3D_result
        }
