#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import glob
import tqdm
import subprocess
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy.spatial import cKDTree

sys.path.append(".")

from Config.pix3d import PIX3DConfig

# preprocess param
del_intermediate_result = True
skip_done = False
processes = 12

# path settings
config = PIX3DConfig()
mesh_folder = os.path.join(config.metadata_path, 'model')
output_root = 'data/pix3d/ldif'
gaps = './external/ldif/gaps/bin/x86_64'
mesh_fusion = 'external/mesh_fusion'
python_bin = sys.executable
skip = ['IKEA_JULES_1.model_-108.706406967_-139.417398691']

# ldif param
scale_norm = 0.25
bbox_half = 0.7
bbox = ' '.join([str(-bbox_half), ] * 3 + [str(bbox_half), ] * 3)
spacing = bbox_half * 2 / 32
print({'bbox_half': bbox_half, 'spacing': spacing})

# mgnet param
neighbors = 30

def read_obj(model_path, flags = ('v')):
    fid = open(model_path, 'r')

    data = {}

    for head in flags:
        data[head] = []

    for line in fid:
        # line = line.strip().split(' ')
        line = re.split('\s+', line.strip())
        if line[0] in flags:
            data[line[0]].append(line[1:])

    fid.close()

    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(float)

    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(float)

    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(float)

    return data

def write_obj(objfile, data):

    with open(objfile, 'w+') as file:
        for item in data['v']:
            file.write('v' + ' %f' * len(item) % tuple(item) + '\n')

        for item in data['f']:
            file.write('f' + ' %s' * len(item) % tuple(item) + '\n')

def det(a):
    return a[0][0]*a[1][1]*a[2][2] + \
        a[0][1]*a[1][2]*a[2][0] + \
        a[0][2]*a[1][0]*a[2][1] - \
        a[0][2]*a[1][1]*a[2][0] - \
        a[0][1]*a[1][0]*a[2][2] - \
        a[0][0]*a[1][2]*a[2][1]

def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    if magnitude == 0.:
        return (0., 0., 0.)
    else:
        return (x/magnitude, y/magnitude, z/magnitude)

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)

def get_area(poly):
    if len(poly) < 3:
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]

    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

def calculate_face_area(data):
    face_areas = []

    for face in data['f']:
        vid_in_face = [int(item.split('/')[0]) for item in face]
        face_area = get_area(data['v'][np.array(vid_in_face) - 1,:3].tolist())
        face_areas.append(face_area)

    return face_areas

def sample_pnts_from_obj(data, n_pnts = 5000, mode = 'uniform'):
    flags = data.keys()

    all_pnts = data['v'][:,:3]

    area_list = np.array(calculate_face_area(data))
    distribution = area_list/np.sum(area_list)

    new_pnts = []
    if mode == 'random':
        random_face_ids = np.random.choice(len(data['f']), n_pnts, replace=True, p=distribution)
        random_face_ids, sample_counts = np.unique(random_face_ids, return_counts=True)

        for face_id, sample_count in zip(random_face_ids, sample_counts):

            face = data['f'][face_id]

            vid_in_face = [int(item.split('/')[0]) for item in face]

            weights = np.diff(np.sort(np.vstack(
                [np.zeros((1, sample_count)),
                 np.random.uniform(0, 1, size=(len(vid_in_face) - 1, sample_count)),
                 np.ones((1, sample_count))]), axis=0), axis=0)

            new_pnt = all_pnts[np.array(vid_in_face) - 1].T.dot(weights)

            if 'vn' in flags:
                nid_in_face = [int(item.split('/')[2]) for item in face]
                new_normal = data['vn'][np.array(nid_in_face)-1].T.dot(weights)
                new_pnt = np.hstack([new_pnt, new_normal])


            new_pnts.append(new_pnt.T)
        return np.vstack(new_pnts)

    for face_idx, face in enumerate(data['f']):
        vid_in_face = [int(item.split('/')[0]) for item in face]

        n_pnts_on_face = distribution[face_idx] * n_pnts

        if n_pnts_on_face < 1:
            continue

        dim = len(vid_in_face)
        npnts_dim = (np.math.factorial(dim - 1)*n_pnts_on_face)**(1/(dim-1))
        npnts_dim = int(npnts_dim)

        weights = np.stack(np.meshgrid(*[np.linspace(0, 1, npnts_dim) for _ in range(dim - 1)]), 0)
        weights = weights.reshape(dim - 1, -1)
        last_column = 1 - weights.sum(0)
        weights = np.vstack([weights, last_column])
        weights = weights[:, last_column >= 0]

        new_pnt = (all_pnts[np.array(vid_in_face) - 1].T.dot(weights)).T

        if 'vn' in flags:
            nid_in_face = [int(item.split('/')[2]) for item in face]
            new_normal = data['vn'][np.array(nid_in_face) - 1].T.dot(weights)
            new_pnt = np.hstack([new_pnt, new_normal])

        new_pnts.append(new_pnt)
    return np.vstack(new_pnts)

def normalize_to_unit_square(points):
    centre = (points.max(0) + points.min(0))/2.
    point_shapenet = points - centre

    scale = point_shapenet.max()
    point_shapenet = point_shapenet / scale

    return point_shapenet, centre, scale

def normalize(input_path, output_folder):
    output_path = os.path.join(output_folder, 'mesh_normalized.obj')

    obj_data = read_obj(input_path, ['v', 'f'])
    obj_data['v'] = normalize_to_unit_square(obj_data['v'])[0]
    write_obj(output_path, obj_data)
    return output_path

def make_watertight(input_path, output_folder):
    output_path = os.path.join(output_folder, 'mesh_orig.obj')

    # convert mesh to off
    off_path = os.path.splitext(output_path)[0] + '.off'
    subprocess.check_output(f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {input_path} -o {off_path}',
                            shell=True)

    # scale mesh
    # app = scale.Scale(
    #     f'--in_file {off_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite'.split(' '))
    # app.run()
    subprocess.check_output(f'{python_bin} {mesh_fusion}/scale.py'
                            f' --in_file {off_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite',
                            shell=True)

    # create depth maps
    # app = fusion.Fusion(
    #     f'--mode=render --in_file {off_path} --out_dir {output_folder} --overwrite'.split(' '))
    # app.run()
    subprocess.check_output(f'xvfb-run -a -s "-screen 0 800x600x24" {python_bin} {mesh_fusion}/fusion.py'
                            f' --mode=render --in_file {off_path} --out_dir {output_folder} --overwrite',
                            shell=True)

    # produce watertight mesh
    depth_path = off_path + '.h5'
    transform_path = os.path.splitext(output_path)[0] + '.npz'
    # app = fusion.Fusion(
    #     f'--mode=fuse --in_file {depth_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite'.split(' '))
    # app.run()
    subprocess.check_output(f'{python_bin} {mesh_fusion}/fusion.py --mode=fuse'
                            f' --in_file {depth_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite',
                            shell=True)

    # # simplify mesh
    # obj_path = os.path.splitext(output_path)[0] + '.obj'
    # app = simplify.Simplification(
    #     f'--in_file={obj_path} --out_dir {output_folder}'.split(' '))
    # app.run()
    # subprocess.check_output(f'xvfb-run -a -s "-screen 0 800x600x24" {python_bin} {mesh_fusion}/simplify.py'
    #                         f' --in_file={obj_path} --out_dir {output_folder}', shell=True)

    os.remove(off_path)
    os.remove(transform_path)
    os.remove(depth_path)
    return output_path


def remove_if_exists(f):
    if os.path.exists(f):
        os.remove(f)


def make_output_folder(mesh_path):
    rel_folder = os.path.relpath(mesh_path, mesh_folder).split('/')
    model_folder = '.'.join(os.path.splitext(mesh_path)[0].split('/')[-2:])
    rel_folder = os.path.join(*rel_folder[:-2], model_folder)
    output_folder = os.path.join(output_root, rel_folder)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def process_mgnet(obj_path, output_folder, ext):
    obj_data = read_obj(obj_path, ['v', 'f'])
    sampled_points = sample_pnts_from_obj(obj_data, 10000, mode='random')
    sampled_points.tofile(os.path.join(output_folder, f'gt_3dpoints.{ext}'))

    tree = cKDTree(sampled_points)
    dists, indices = tree.query(sampled_points, k=neighbors)
    densities = np.array([max(dists[point_set, 1]) ** 2 for point_set in indices])
    densities.tofile(os.path.join(output_folder, f'densities.{ext}'))


def process_mesh(mesh_path):
    output_folder = make_output_folder(mesh_path)
    mesh_name = os.path.basename(output_folder)
    if mesh_name in skip:
        print(f"skipping {mesh_name}")
        return
    if skip_done and os.path.exists(f'{output_folder}/uniform_points.sdf'):
        return

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
    os.system(f'{gaps}/msh2msh {watertight_ply} {scaled_ply} -scale_by_pca -translate_by_centroid'
              f' -scale {scale_norm} -debug_matrix {output_folder}/orig_to_gaps.txt')

    # Step 1) Generate the coarse inside/outside grid:
    os.system(f'{gaps}/msh2df {scaled_ply} {output_folder}/coarse_grid.grd'
              f' -bbox {bbox} -border 0 -spacing {spacing} -estimate_sign')

    # Step 2) Generate the near surface points:
    os.system(f'{gaps}/msh2pts {scaled_ply} {output_folder}/nss_points.sdf'
              f' -near_surface -max_distance {spacing} -num_points 100000 -binary_sdf')

    # Step 3) Generate the uniform points:
    os.system(f'{gaps}/msh2pts {scaled_ply} {output_folder}/uniform_points.sdf'
              f' -uniform_in_bbox -bbox {bbox} -npoints 100000 -binary_sdf')

    # Step 4) Generate surface points for MGNet:
    process_mgnet(watertight_obj, output_folder, 'mgn')
    process_mgnet(normalized_obj, output_folder, 'org')

    if del_intermediate_result:
        remove_if_exists(normalized_obj)
        remove_if_exists(watertight_obj)
        remove_if_exists(scaled_ply)


def process_img(sample):
    output_folder = make_output_folder(os.path.join(config.metadata_path, sample['model']))
    img_name = os.path.splitext(os.path.split(sample['img'])[1])[0]
    output_path = os.path.join(output_folder, img_name + '.npy')
    if not skip_done or not os.path.exists(output_path):
        img = np.array(Image.open(os.path.join(config.metadata_path, sample['img'])).convert('RGB'))
        img = img[sample['bbox'][1]:sample['bbox'][3], sample['bbox'][0]:sample['bbox'][2]]
        np.save(output_path, img)
    img_name = os.path.splitext(os.path.split(sample['mask'])[1])[0]
    output_path = os.path.join(output_folder, img_name + '_mask.npy')
    if not skip_done or not os.path.exists(output_path):
        img = np.array(Image.open(os.path.join(config.metadata_path, sample['mask'])).convert('L'))
        img = img[sample['bbox'][1]:sample['bbox'][3], sample['bbox'][0]:sample['bbox'][2]]
        np.save(output_path, img)




if __name__ == '__main__':
    print('Processing imgs...')
    with open(config.metadata_file, 'r') as file:
        metadata = json.load(file)

    with open(config.train_split, 'r') as file:
        splits = json.load(file)

    with open(config.test_split, 'r') as file:
        splits += json.load(file)

    ids = [int(os.path.basename(file).split('.')[0]) for file in splits if 'flipped' not in file]
    samples = [metadata[id] for id in ids]

    if processes:
        with Pool(processes=processes) as p:
            r = list(tqdm.tqdm(p.imap(process_img, samples), total=len(samples)))
    else:
        for sample in tqdm.tqdm(samples):
            process_img(sample)

    print('Processing meshs...')
    mesh_paths = glob.glob(os.path.join(mesh_folder, '*', '*', '*.obj'))

    if processes:
        with Pool(processes=processes) as p:
            r = list(tqdm.tqdm(p.imap(process_mesh, mesh_paths), total=len(mesh_paths)))
    else:
        for mesh_path in tqdm.tqdm(mesh_paths):
            process_mesh(mesh_path)
