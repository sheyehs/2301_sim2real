import open3d as o3d
import yaml
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

model_dir = '/home/sim2real/code_Yaoen/models'

def f(pcd_path):
    print('extract model point cloud from:', pcd_path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = np.array(pcd.points)
    # pcd_index = np.random.choice(pcd.shape[0], 1000)
    # pcd = pcd[pcd_index]
    diameter = 0
    for i in tqdm(range(pcd.shape[0] - 1)):
        x = (pcd[i][0], pcd[i][1], pcd[i][2])
        for j in range(i + 1, pcd.shape[0]):
            y = (pcd[j][0], pcd[j][1], pcd[j][2])
            d = ((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2) ** 0.5
            diameter = max(diameter, d)

    part_name = pcd_path.split('/')[-1].split('.')[0]
    print(f'diameter of {part_name} is {diameter}')

    info = {str(part_name): str(diameter)}

    with open(os.path.join(model_dir, f'_{part_name}_diameters.yaml'), 'w') as file:
        yaml.dump(info, file)

if __name__ == '__main__':
    if 0:
        pcd_list = glob(os.path.join(model_dir, '*.master.ply'))
        with Pool(cpu_count()) as p:
            p.map(f, pcd_list)

    if 1:  # merge seperate diameter yamls
        yaml_list = glob(os.path.join(model_dir, '_*.yaml'))
        diameters = {}
        for yaml_file in yaml_list:
            with open(yaml_file, 'r') as file:
                info = yaml.safe_load(file)
            print(info)
            diameters.update(info)
        with open(os.path.join(model_dir, 'diameters.yaml'), 'w') as file:
            yaml.dump(diameters, file)


