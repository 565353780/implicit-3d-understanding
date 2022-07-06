#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import tqdm
import subprocess
import numpy as np
from PIL import Image
from multiprocessing import Pool

sys.path.append(".")

from Config.configs import PIX3DConfig

from Method.preprocess import \
    normalize, make_watertight, remove_if_exists, process_mgnet

class PreProcesser(object):
    def __init__(self, config):
        self.del_intermediate_result = True
        self.skip_done = False
        self.processes = 12

        self.scale_norm = 0.25
        self.bbox_half = 0.7

        self.gaps_folder_path = "./external/ldif/gaps/bin/x86_64/"
        self.mesh_fusion_folder_path = ".external/mesh_fusion/"
        self.python_bin = sys.executable

        self.config = config

        self.bbox = ' '.join([str(-self.bbox_half), ] * 3 + [str(self.bbox_half), ] * 3)
        self.spacing = self.bbox_half * 2 / 32
        return

class PIX3DPreProcesser(PreProcesser):
    def __init__(self, config):
        super(PIX3DPreProcesser, self).__init__(config)

        self.mesh_folder = self.config.metadata_path + "model"
        self.output_root = self.config.root_path + "ldif"
        self.skip = ['IKEA_JULES_1.model_-108.706406967_-139.417398691']

        self.neighbors = 30
        return

    def make_output_folder(self, mesh_path):
        rel_folder = os.path.relpath(mesh_path, self.mesh_folder).split('/')
        model_folder = '.'.join(os.path.splitext(mesh_path)[0].split('/')[-2:])
        rel_folder = os.path.join(*rel_folder[:-2], model_folder)
        output_folder = os.path.join(self.output_root, rel_folder)
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def processImage(self, sample):
        output_folder = self.make_output_folder(os.path.join(self.config.metadata_path, sample['model']))
        img_name = os.path.splitext(os.path.split(sample['img'])[1])[0]
        output_path = os.path.join(output_folder, img_name + '.npy')
        if not self.skip_done or not os.path.exists(output_path):
            img = np.array(Image.open(os.path.join(self.config.metadata_path, sample['img'])).convert('RGB'))
            img = img[sample['bbox'][1]:sample['bbox'][3], sample['bbox'][0]:sample['bbox'][2]]
            np.save(output_path, img)
        img_name = os.path.splitext(os.path.split(sample['mask'])[1])[0]
        output_path = os.path.join(output_folder, img_name + '_mask.npy')
        if not self.skip_done or not os.path.exists(output_path):
            img = np.array(Image.open(os.path.join(self.config.metadata_path, sample['mask'])).convert('L'))
            img = img[sample['bbox'][1]:sample['bbox'][3], sample['bbox'][0]:sample['bbox'][2]]
            np.save(output_path, img)
        return True

    def processMesh(self, mesh_path):
        output_folder = self.make_output_folder(mesh_path)
        mesh_name = os.path.basename(output_folder)
        if mesh_name in self.skip:
            print(f"skipping {mesh_name}")
            return True
        if self.skip_done and os.path.exists(f'{output_folder}/uniform_points.sdf'):
            return True

        # Step 0) Normalize and watertight the mesh before applying all other operations.
        normalized_obj = normalize(mesh_path, output_folder)
        watertight_obj = make_watertight(normalized_obj, output_folder)

        # conver mesh to ply
        normalized_ply = os.path.splitext(normalized_obj)[0] + '.ply'
        subprocess.check_output(
            f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {normalized_obj} -o {normalized_ply}',
            shell=True)
        watertight_ply = os.path.splitext(watertight_obj)[0] + '.ply'
        subprocess.check_output(
            f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {watertight_obj} -o {watertight_ply}',
            shell=True)

        scaled_ply = os.path.join(output_folder, 'scaled_watertight.ply')
        os.system(f'{self.gaps_folder_path}/msh2msh {watertight_ply} {scaled_ply} -scale_by_pca -translate_by_centroid'
                  f' -scale {self.scale_norm} -debug_matrix {output_folder}/orig_to_gaps.txt')

        # Step 1) Generate the coarse inside/outside grid:
        os.system(f'{self.gaps_folder_path}/msh2df {scaled_ply} {output_folder}/coarse_grid.grd'
                  f' -bbox {self.bbox} -border 0 -spacing {self.spacing} -estimate_sign')

        # Step 2) Generate the near surface points:
        os.system(f'{self.gaps_folder_path}/msh2pts {scaled_ply} {output_folder}/nss_points.sdf'
                  f' -near_surface -max_distance {self.spacing} -num_points 100000 -binary_sdf')

        # Step 3) Generate the uniform points:
        os.system(f'{self.gaps_folder_path}/msh2pts {scaled_ply} {output_folder}/uniform_points.sdf'
                  f' -uniform_in_bbox -bbox {self.bbox} -npoints 100000 -binary_sdf')

        # Step 4) Generate surface points for MGNet:
        process_mgnet(watertight_obj, output_folder, 'mgn', self.neighbors)
        process_mgnet(normalized_obj, output_folder, 'org', self.neighbors)

        if self.del_intermediate_result:
            remove_if_exists(normalized_obj)
            remove_if_exists(watertight_obj)
            remove_if_exists(scaled_ply)
        return True

    def processAllImage(self):
        print('Processing imgs...')
        with open(self.config.metadata_file, 'r') as file:
            metadata = json.load(file)

        with open(self.config.train_split, 'r') as file:
            splits = json.load(file)

        with open(self.config.test_split, 'r') as file:
            splits += json.load(file)

        ids = [int(os.path.basename(file).split('.')[0]) for file in splits if 'flipped' not in file]
        samples = [metadata[id] for id in ids]

        with Pool(self.processes) as p:
            r = list(tqdm.tqdm(p.imap(self.processImage, samples), total=len(samples)))
        return True

    def processAllMesh(self):
        mesh_paths = glob.glob(os.path.join(self.mesh_folder, '*', '*', '*.obj'))

        with Pool(self.processes) as p:
            r = list(tqdm.tqdm(p.imap(self.processMesh, mesh_paths), total=len(mesh_paths)))
        return True

    def process(self):
        if not self.processAllImage():
            print("[ERROR][PIX3DPreProcesser::process]")
            print("\t processAllImage failed!")
            return False
        if not self.processAllMesh():
            print("[ERROR][PIX3DPreProcesser::process]")
            print("\t processAllMesh failed!")
            return False
        return True

def demo():
    config = PIX3DConfig()
    PreProcesser = PIX3DPreProcesser

    preprocesser = PreProcesser(config)
    preprocesser.process()
    return True

if __name__ == '__main__':
    demo()

