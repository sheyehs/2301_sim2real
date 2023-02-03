import numpy as np
import open3d as o3d
import os
import cv2
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import h5py


data_dir = '/scratch/gc2720/2301_sim2real/2022-11-15'
hdf5_path = '/scratch/gc2720/2301_sim2real/data.hdf5'
company_list = ["SongFeng", "Toyota","ZSRobot"]
downsample_factor = 3  # (1200, 1920) -> (400, 640)

def extract_pcd(part_dir):
    part_name = part_dir.split('/')[-1]
    pcd_path = os.path.join(part_dir, 'part_info', f'{part_name}.master.ply')
    print('extract model point cloud from:', pcd_path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = np.array(pcd.points)
    print('model point cloud shape:', pcd.shape)
    ones = np.ones((pcd.shape[0], 1))
    pcd_one = np.concatenate([pcd, ones], axis=1)
    return pcd_one

def filter_nearest_pixels(pixels):
    # shape: (3 or 4, numbers of pixels)
    # by rows: i, j, depth(, instance_index)
    order = np.argsort(pixels[2])
    pixels = pixels[:, order]
    _, keep = np.unique(pixels[:2], axis=1, return_index=True)  # drop duplicates and keep the first element (smallest depth)
    pixels = pixels[:, keep]
    return pixels


def extract_instances_from_one_image(image_name, condition_dir, condition_grp, pcd_one):
    condition_grp = h.require_group(condition_grp)
    image_grp = condition_grp.require_group(image_name)
    depth_path = os.path.join(condition_dir, 'depth', f'{image_name}.png')  # keep same
    color_path = os.path.join(condition_dir, 'images', f'{image_name}.jpg')  # keep same
    param_path = os.path.join(condition_dir, 'images', f'{image_name}.json')
    label_path = os.path.join(condition_dir, 'labels', f'{image_name}.npy')
    
    # read json
    with open(param_path) as f:
        params = json.load(f)
    image_shape = np.array([params['height'], params['width']], dtype=int)
    K = np.array(params['K']).reshape((3,3))
    # downsample shape and K
    image_shape = (image_shape / downsample_factor).astype(int)
    K /= downsample_factor
    # save shape and K
    image_grp.create_dataset('height', data=image_shape[0])
    image_grp.create_dataset('width', data=image_shape[1])
    image_grp.create_dataset('K', data=K)

    # read images
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    # downsample
    depth = cv2.resize(depth, (image_shape[1], image_shape[0]))  # first width, then height
    color = cv2.resize(color, (image_shape[1], image_shape[0]))
    # save images
    image_grp.create_dataset('depth', data=depth)
    image_grp.create_dataset('color', data=color)

    # read numpy
    labels = np.load(label_path)
    n_instances = labels.shape[0]
    pixels_list = []
    for i, pose in enumerate(labels):

        instance_name = f'{i:02}'
        instance_grp = image_grp.require_group(instance_name)

        instance_grp.create_dataset('R', data=pose[:3, :3])
        instance_grp.create_dataset('t', data=pose[:3, 3])

        points = np.matmul(pose, pcd_one.transpose())  # transform model points
        normalized_points = points[:3] / points[2]  # normalized plane
        pixels = np.floor(np.matmul(K, normalized_points)[:2])  # project model points to image
        pixels[[0,1]] = pixels[[1,0]]  # swap (x ,y) to (y, x)
        pixels = np.concatenate([pixels, points[[2]]], axis=0)  # concat depth

        keep = (pixels[0]>=0) & (pixels[0]<image_shape[0]) & (pixels[1]>=0) & (pixels[1]<image_shape[1])
        pixels = pixels[:, keep]

        instance_grp.create_dataset('y_min', data=pixels[0].min())
        instance_grp.create_dataset('y_max', data=pixels[0].max())
        instance_grp.create_dataset('x_min', data=pixels[1].min())
        instance_grp.create_dataset('x_max', data=pixels[1].max())

        pixels = filter_nearest_pixels(pixels)

        instance_index = np.full((1, pixels.shape[1]), i)
        pixels = np.concatenate([pixels, instance_index], axis=0)

        pixels_list.append(pixels)
        # instance_grp.create_dataset('mask', data=mask)

    pixels_all = np.concatenate(pixels_list, axis=1)
    pixels_all = filter_nearest_pixels(pixels_all)
    
    for i in range(n_instances):
        pixels = pixels_all[:2, pixels_all[3] == i].astype(int)
        flat_mask = np.zeros(image_shape, dtype=bool).ravel()
        flat_index_array = np.ravel_multi_index(pixels, image_shape)
        flat_mask[flat_index_array] = True
        mask = flat_mask.reshape(image_shape)

        instance_name = f'{i:02}'
        image_grp.create_dataset(f'{instance_name}/mask', data=mask)


if __name__ == '__main__':
    h = h5py.File(hdf5_path, "a")
    for company_name in company_list:
        company_grp = h.require_group(company_name)
        company_dir = os.path.join(data_dir, company_name)
        part_list = os.listdir(company_dir)
        for part_name in part_list:
            part_grp = company_grp.require_group(part_name)
            part_dir = os.path.join(company_dir, part_name)
            condition_list = [c for c in os.listdir(part_dir) if c.endswith('synthetic')]
            pcd_one = extract_pcd(part_dir)
            for condition_name in tqdm(condition_list):
                condition_grp = part_grp.require_group(condition_name)
                condition_dir = os.path.join(part_dir, condition_name)
                image_list = [i.split('.')[0] for i in os.listdir(os.path.join(condition_dir, 'depth'))]
                print(f"found {len(image_list)} images in {condition_dir}")
                image_list.sort()

                # f = partial(
                #     extract_instances_from_one_image, 
                #     condition_dir=condition_dir, condition_grp=condition_grp.name, pcd_one=pcd_one
                #     # pool cannot pass h5py object, so pass its name and retieve back whithin the function
                # )
                # with Pool(cpu_count()) as p:
                #     p.map(f, image_list)
                
                for image_name in tqdm(image_list):
                    extract_instances_from_one_image(image_name, condition_dir, condition_grp.name, pcd_one)

    h.close()