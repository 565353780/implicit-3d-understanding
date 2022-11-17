#!/usr/bin/env python
# -*- coding: utf-8 -*-

from models.registers import MODULES, LOSSES
from models.network import BaseNetwork
import torch
from torch import nn
from configs.data_config import obj_cam_ratio
from external.ldif.representation.structured_implicit_function import StructuredImplicit

from pose_detect.Config.ldif import LDIF_CONFIG

from pose_detect.Model.pose_net import PoseNet
from pose_detect.Model.bdb_3d.bdb_3d_net import Bdb3DNet
from pose_detect.Model.ldif.ldif import LDIF

from pose_detect.Loss.pose_loss import PoseLoss
from pose_detect.Loss.det_loss import DetLoss
from pose_detect.Loss.ldif_loss import LDIFReconLoss


class TOTAL3D(BaseNetwork):

    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        self.layout_estimation = PoseNet(cfg)
        self.layout_estimation_loss = PoseLoss(cfg.config)

        self.object_detection = Bdb3DNet(cfg)
        self.object_detection_loss = DetLoss(cfg.config)

        self.mesh_reconstruction = LDIF(LDIF_CONFIG, "test")
        self.mesh_reconstruction_loss = LDIFReconLoss(cfg.config)

        phase_names = []
        #  phase_names += ['mesh_reconstruction']
        phase_names += ['output_adjust']
        '''load network blocks'''
        for phase_name in phase_names:
            net_spec = cfg.config['model'][phase_name]
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = cfg.config['optimizer']
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)
            '''load corresponding loss functions'''
            setattr(
                self, phase_name + '_loss',
                LOSSES.get(self.cfg.config['model'][phase_name]['loss'],
                           'Null')(self.cfg.config['model'][phase_name].get(
                               'weight', 1), cfg.config))

        setattr(self, 'joint_loss', LOSSES.get('JointLoss', 'Null')(1))

        self.layout_estimation = nn.DataParallel(self.layout_estimation)
        self.mesh_reconstruction = nn.DataParallel(self.mesh_reconstruction)
        self.freeze_modules(cfg)
        return

    def get_extra_results(self, all_output):
        extra_results = {}

        structured_implicit = all_output['structured_implicit']
        in_coor_min = structured_implicit.all_centers.min(dim=1)[0]
        in_coor_max = structured_implicit.all_centers.max(dim=1)[0]

        obj_center = (in_coor_max + in_coor_min) / 2.
        obj_center[:, 2] *= -1
        obj_coef = (in_coor_max - in_coor_min) / 2.

        extra_results.update({'obj_center': obj_center, 'obj_coef': obj_coef})
        return extra_results

    def forward(self, data):
        all_output = {}

        pitch_reg_result, roll_reg_result, \
        pitch_cls_result, roll_cls_result, \
        lo_ori_reg_result, lo_ori_cls_result, \
        lo_centroid_result, lo_coeffs_result, a_features = self.layout_estimation(data['image'])

        layout_output = {
            'pitch_reg_result': pitch_reg_result,
            'roll_reg_result': roll_reg_result,
            'pitch_cls_result': pitch_cls_result,
            'roll_cls_result': roll_cls_result,
            'lo_ori_reg_result': lo_ori_reg_result,
            'lo_ori_cls_result': lo_ori_cls_result,
            'lo_centroid_result': lo_centroid_result,
            'lo_coeffs_result': lo_coeffs_result,
            'lo_afeatures': a_features
        }
        all_output.update(layout_output)

        size_reg_result, \
        ori_reg_result, ori_cls_result, \
        centroid_reg_result, centroid_cls_result, \
        offset_2D_result, a_features, \
        r_features, a_r_features = self.object_detection(data['patch'], data['size_cls'], data['g_features'],
                                                         data['split'], data['rel_pair_counts'])
        object_output = {
            'size_reg_result': size_reg_result,
            'ori_reg_result': ori_reg_result,
            'ori_cls_result': ori_cls_result,
            'centroid_reg_result': centroid_reg_result,
            'centroid_cls_result': centroid_cls_result,
            'offset_2D_result': offset_2D_result,
            'odn_afeature': a_features,
            'odn_rfeatures': r_features,
            'odn_arfeatures': a_r_features
        }
        all_output.update(object_output)

        # predict meshes
        if self.cfg.config['mode'] == 'train':
            data['img'] = data['patch_for_mesh']
            data['cls'] = data['cls_codes']
            mesh_output = self.mesh_reconstruction(data)
        else:
            data['img'] = data['patch_for_mesh']
            data['cls'] = data['cls_codes']
            mesh_output = self.mesh_reconstruction(data)
            out_points = mesh_output.get('mesh_coordinates_results', [None])
            out_faces = mesh_output.get('faces', None)
            mesh_output.update({
                'meshes': out_points[-1],
                'out_faces': out_faces
            })

        if 'structured_implicit' in mesh_output:
            mesh_output['structured_implicit'] = StructuredImplicit(
                config=self.cfg.config, **mesh_output['structured_implicit'])

        if mesh_output.get('meshes') is not None:
            if isinstance(mesh_output['meshes'], list):
                for m in mesh_output['meshes']:
                    m[2, :] *= -1
            elif mesh_output['meshes'] is not None:
                mesh_output['meshes'][:, 2, :] *= -1
        mesh_output['mgn'] = self.mesh_reconstruction
        all_output.update(mesh_output)

        all_output.update(self.get_extra_results(all_output))

        if hasattr(self, 'output_adjust'):
            input = all_output.copy()
            input['size_cls'] = data['size_cls']
            input['cls_codes'] = data['cls_codes']
            input['g_features'] = data['g_features']
            input['bdb2D_pos'] = data['bdb2D_pos']
            input['K'] = data['K']
            input['split'] = data['split']
            input['rel_pair_counts'] = data['rel_pair_counts']
            refined_output = self.output_adjust(input)
            all_output.update(refined_output)

        return all_output

    def loss(self, est_data, gt_data):
        '''
        calculate loss of est_out given gt_out.
        '''
        loss_weights = self.cfg.config.get('loss_weights', {})
        if self.cfg.config[self.cfg.config['mode']]['phase'] in [
                'layout_estimation', 'joint'
        ]:
            layout_loss, layout_results = self.layout_estimation_loss(
                est_data, gt_data, self.cfg.bins_tensor)
            layout_loss_weighted = {
                k: v * loss_weights.get(k, 1.0)
                for k, v in layout_loss.items()
            }
            total_layout_loss = sum(layout_loss_weighted.values())
            total_layout_loss_unweighted = sum(
                [v.detach() for v in layout_loss.values()])
            for key, value in layout_loss.items():
                layout_loss[key] = value.item()
        if self.cfg.config[self.cfg.config['mode']]['phase'] in [
                'object_detection', 'joint'
        ]:
            object_loss = self.object_detection_loss(est_data, gt_data)
            object_loss_weighted = {
                k: v * loss_weights.get(k, 1.0)
                for k, v in object_loss.items()
            }
            total_object_loss = sum(object_loss_weighted.values())
            total_object_loss_unweighted = sum(
                [v.detach() for v in object_loss.values()])
            for key, value in object_loss.items():
                object_loss[key] = value.item()
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['joint']:
            joint_loss, extra_results = self.joint_loss(
                est_data, gt_data, self.cfg.bins_tensor, layout_results)
            joint_loss_weighted = {
                k: v * loss_weights.get(k, 1.0)
                for k, v in joint_loss.items()
            }
            mesh_loss = self.mesh_reconstruction_loss(est_data, gt_data,
                                                      extra_results)
            mesh_loss_weighted = {
                k: v * loss_weights.get(k, 1.0)
                for k, v in mesh_loss.items()
            }

            total_joint_loss = sum(joint_loss_weighted.values()) + sum(
                mesh_loss_weighted.values())
            total_joint_loss_unweighted = \
                sum([v.detach() for v in joint_loss.values()]) \
                + sum([v.detach() if isinstance(v, torch.Tensor) else v for v in mesh_loss.values()])
            for key, value in mesh_loss.items():
                mesh_loss[key] = float(value)
            for key, value in joint_loss.items():
                joint_loss[key] = value.item()

        if self.cfg.config[
                self.cfg.config['mode']]['phase'] == 'layout_estimation':
            return {
                'total': total_layout_loss,
                **layout_loss, 'total_unweighted': total_layout_loss_unweighted
            }
        if self.cfg.config[
                self.cfg.config['mode']]['phase'] == 'object_detection':
            return {
                'total': total_object_loss,
                **object_loss, 'total_unweighted': total_object_loss_unweighted
            }
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'joint':
            total3d_loss = total_object_loss + total_joint_loss + obj_cam_ratio * total_layout_loss
            total3d_loss_unweighted = total_object_loss_unweighted + total_joint_loss_unweighted\
                                      + obj_cam_ratio * total_layout_loss_unweighted
            return {
                'total': total3d_loss,
                **layout_loss,
                **object_loss,
                **mesh_loss,
                **joint_loss, 'total_unweighted': total3d_loss_unweighted
            }
        else:
            raise NotImplementedError
