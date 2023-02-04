import os
import random
import h5py
from sklearn.model_selection import train_test_split

use_ratio = 0.025
test_ratio = 0.1
data_path = '../data.hdf5'
oroot = '../train_test/'

h = h5py.File(data_path, 'r')

for company_name in h:
    company = h[company_name]
    for part_name in company:
        opath = os.path.join(oroot, part_name)
        os.makedirs(opath, exist_ok=True)
        part = company[part_name]
        for condition_name in part:
            instance_list = []
            condition = part[condition_name]
            print(f'found {len(condition.keys())} images in {condition.name}')
            for image_name in condition:
                image = condition[image_name]
                for instance_name in image:
                    if instance_name in ['height', 'width', 'K', 'depth', 'color']: continue
                    name = image_name + '_' + instance_name
                    instance_list.append(name)

            n_use = int(use_ratio * len(instance_list))
            instances = random.sample(instance_list, n_use)
            train, test = train_test_split(instances, test_size=test_ratio)

            with open(os.path.join(opath, 'test.txt'), 'w') as f:
                f.write('\n'.join(test))
            with open(os.path.join(opath, 'train.txt'), 'w') as f:
                f.write('\n'.join(train))

h.close()