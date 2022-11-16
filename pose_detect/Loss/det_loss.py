#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from configs.data_config import cls_reg_ratio


class DetLoss(object):

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

    def __call__(self, est_data, gt_data):
        # calculate loss
        size_reg_loss = self.reg_criterion(est_data['size_reg_result'],
                                           gt_data['size_reg']) * cls_reg_ratio
        ori_cls_loss, ori_reg_loss = self.cls_reg_loss(
            est_data['ori_cls_result'], gt_data['ori_cls'],
            est_data['ori_reg_result'], gt_data['ori_reg'])
        centroid_cls_loss, centroid_reg_loss = self.cls_reg_loss(
            est_data['centroid_cls_result'], gt_data['centroid_cls'],
            est_data['centroid_reg_result'], gt_data['centroid_reg'])
        offset_2D_loss = self.reg_criterion(est_data['offset_2D_result'],
                                            gt_data['offset_2D'])

        return {
            'size_reg_loss': size_reg_loss,
            'ori_cls_loss': ori_cls_loss,
            'ori_reg_loss': ori_reg_loss,
            'centroid_cls_loss': centroid_cls_loss,
            'centroid_reg_loss': centroid_reg_loss,
            'offset_2D_loss': offset_2D_loss
        }
