import h5py
import os
from matplotlib import pyplot as plt
import numpy as np

company = 'ZSRobot'
part = '6010022CSV'
condition = '2022-05-19-rvbust_synthetic'
image = '002453'
instance = '10'
hdf5_path = '/scratch/gc2720/2301_sim2real/data.hdf5'

"""
data.hdf5 hierarchy
company | part  | model_info
                | condition | image | height
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
print('find company:', h.keys())
print('find part:', h[f'{company}'].keys())
print('find condition:', h[f'{company}/{part}'].keys())
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