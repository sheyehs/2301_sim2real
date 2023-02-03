import numpy as np 

path = './2022-11-15/SongFeng/SongFeng_332/2022-04-23-rvbust_normal_lighting_lux134/labels/000000.npy'
# path = '2022-11-15/SongFeng/SongFeng_332/2022-04-28-rvbust_synthetic/labels/000018.npy'
x = np.load(path)
print(path)
print(x)
print("shape:", x.shape)