import h5py

f = h5py.File('./datasets_pose_estimation_yaoen/data/01/mask.hdf5')

print(f['000000'].keys())