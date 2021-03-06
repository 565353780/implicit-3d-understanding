#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import trimesh
import numpy as np
from skimage import measure

def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1])):
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4, dtype=np.float32)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    coords = coords.reshape(3, resX, resY, resZ).astype(np.float32)
    return coords, coords_matrix

def batch_eval(points, eval_func, num_samples=512 * 512 * 512, batch_size=1):
    num_pts = points.shape[1]
    sdf = np.zeros([batch_size, num_pts], dtype=np.float32)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf_batch = eval_func(points[:, i * num_samples:i * num_samples + num_samples])
        sdf[:, i * num_samples:i * num_samples + num_samples] = sdf_batch
    if num_pts % num_samples:
        sdf_batch = eval_func(points[:, num_batches * num_samples:])
        sdf[:, num_batches * num_samples:] = sdf_batch
    return sdf

def eval_grid_octree(coords, eval_func,
                     init_resolution=16, threshold=0.01,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution, dtype=np.float32)

    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        test_mask = np.logical_and(grid_mask, dirty)
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)[0]
        dirty[test_mask] = False

        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + reso]
                    v2 = sdf[x, y + reso, z]
                    v3 = sdf[x, y + reso, z + reso]
                    v4 = sdf[x + reso, y, z]
                    v5 = sdf[x + reso, y, z + reso]
                    v6 = sdf[x + reso, y + reso, z]
                    v7 = sdf[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    if (v_max - v_min) < threshold:
                        sdf[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2
    return sdf.reshape(-1, *resolution)

def reconstruction(structured_implicit,
                   resolution, b_min, b_max,
                   num_samples=10000,
                   marching_cube=True):
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max)

    def eval_func(points, structured_implicit):
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=structured_implicit.device).float()
        samples = samples.transpose(-1, -2)
        samples = samples.expand(structured_implicit.batch_size, -1, -1)
        pred = structured_implicit.class_at_samples(samples, apply_class_transfer=False)[0][..., 0]
        return pred.detach().cpu().numpy()

    sdf = []
    for s in structured_implicit.unbind():
        sdf.append(eval_grid_octree(coords, lambda p:eval_func(p, s), num_samples=num_samples).squeeze())
    sdf = np.stack(sdf)

    if not marching_cube:
        return sdf, mat

    mesh = []
    for s in sdf:
        try:
            verts, faces, _, _ = measure.marching_cubes(s, -0.07)
            verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
            verts = verts.T
            mesh.append(trimesh.Trimesh(vertices=verts, faces=faces))
        except (ValueError, RuntimeError) as e:
            print('Failed to extract mesh with error %s. Setting to unit sphere.' % repr(e))
            mesh.append(trimesh.primitives.Sphere(radius=0.5))
    return mesh

