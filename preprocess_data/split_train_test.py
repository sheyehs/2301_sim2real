import os
import glob
import random
import h5py
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

use_ratio = 0.025 * 0.1
test_ratio = 0.1
data_dir = '/scratch/gc2720/2301_sim2real/datasets_pose_estimation_yaoen/data/'
folders = glob.glob(data_dir + '*')

for folder in tqdm(folders):
    print(folder)
    instances = []
    images = glob.glob(os.path.join(folder, 'label', '*.json'))
    for image in images:
        i = image.split('/')[-1].split('.')[0]
        with open(image, 'r') as f_json:
            js = json.load(f_json)['instances'].keys()
            for j in js:
                instances.append(i+'_'+j)

    # with json.load(os.path.join(folder, 'mask.hdf5'), 'r') as f_h5:
    #     for i in f_h5.keys():
    #         if len(f_h5[i].keys()) == 0:
    #             continue
    #         for j in f_h5[i].keys():
    #             instances.append('i'+'_'+'j')

    # instances = glob.glob(os.path.join(folder, 'label', '*.json'))
    # instances = [i.split('/')[-1].split('.')[0] for i in instances]
    n_use = int(use_ratio * len(instances))
    instances = random.sample(instances, n_use)
    # instances_train = [i for i in instances if i not in instances_test]
    train, test = train_test_split(instances, test_size=test_ratio)

    with open(os.path.join(folder, 'test.txt'), 'w') as f:
        f.write('\n'.join(test))
    with open(os.path.join(folder, 'train.txt'), 'w') as f:
        f.write('\n'.join(train))