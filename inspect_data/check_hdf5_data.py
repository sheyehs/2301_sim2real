import h5py
import os
from matplotlib import pyplot as plt
import numpy as np

hdf5_path = '/scratch/gc2720/2301_sim2real/data_real_new.hdf5'

if 0:
    company = 'ZSRobot'
    part = '6010022CSV'
    condition = '2022-02-09-photoneo_bright_lighting_lux3000'
    image = '000044'
    instance = '10'
if 0:
    company = 'Toyota'
    part = '21092302'
    condition = '2022-04-18-rvbust_normal_lighting_lux200'
    image = '000021'
    instance = '02'
if 1:
    company = 'SongFeng'
    part = 'SongFeng_306'
    condition = '2022-04-21-rvbust_bright_lighting_lux1200'
    image = '000014'
    instance = '05'
"""
data.hdf5 hierarchy
company | part  | condition | image | height
                                    | width
                                    | K
                                    | depth
                                    | color
                                    | instance  | mask
                                                | R
                                                | t
                                                | y_min, y_max, x_min, x_max

"""


print('file size:', os.path.getsize(hdf5_path)/1024**3, 'GB')

os.makedirs('./check_hdf5_image', exist_ok=True)

h = h5py.File(hdf5_path, "r")

print()
print('recursive listing...')
print('find company:', list(h.keys()))
print('find part:', list(h[f'{company}'].keys()))
print('find condition:', list(h[f'{company}/{part}'].keys()))
print('find number of images:', len(h[f'{company}/{part}/{condition}'].keys()))
print('find number of instances:', len(h[f'{company}/{part}/{condition}/{image}'].keys()))
print()

print('image info...')
print('image height:', np.array(h.get(f'{company}/{part}/{condition}/{image}/height')), 
    'image width:', np.array(h.get(f'{company}/{part}/{condition}/{image}/width')))
print('K', h[f'{company}/{part}/{condition}/{image}/K'][:])
x = h[f'{company}/{part}/{condition}/{image}/{instance}/mask'][:]
print('mask shape:', x.shape)
plt.imsave('./check_hdf5_image/mask.png', x)
x = h[f'{company}/{part}/{condition}/{image}/color'][:]
print('color shape:', x.shape)
plt.imsave('./check_hdf5_image/color.png', x)
x = h[f'{company}/{part}/{condition}/{image}/depth'][:]
print('depth shape:', x.shape)
plt.imsave('./check_hdf5_image/depth.png', x)
print()

print('instance info...')
print("pose: ", h[f'{company}/{part}/{condition}/{image}/{instance}/R'][:], 
    h[f'{company}/{part}/{condition}/{image}/{instance}/t'][:])
print('y_min, y_max, x_min, x_max:', 
        np.array(h[f'{company}/{part}/{condition}/{image}/{instance}/y_min']),
        np.array(h[f'{company}/{part}/{condition}/{image}/{instance}/y_max']),
        np.array(h[f'{company}/{part}/{condition}/{image}/{instance}/x_min']),
        np.array(h[f'{company}/{part}/{condition}/{image}/{instance}/x_max']))

h.close()