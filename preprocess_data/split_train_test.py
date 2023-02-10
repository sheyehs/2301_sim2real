import os
import random
import h5py

i_mode = 2
out_dir = '../split'

#############################################

mode_list = [
    'train_test',
    'eval',
    'eval_on_real'
]

mode = mode_list[i_mode]

"""
train mode splits train and test data used in train.py
eval mode chooses data for eval.py which visualizes the predicted position onto original image.  
"""

if mode == 'train_test':
    use_ratio = 0.025
    test_ratio = 0.1
    data_path = '../data.hdf5'
    from sklearn.model_selection import train_test_split
elif mode == 'eval':
    use_ratio = 0.0025
    data_path = '../data.hdf5'
elif mode == 'eval_on_real':
    use_ratio = 1
    data_path = '../data_real_new.hdf5'

h = h5py.File(data_path, 'r')

for company_name in h:
    company = h[company_name]
    for part_name in company:
        part = company[part_name]
        out_path = os.path.join(out_dir, part_name)
        os.makedirs(out_path, exist_ok=True)
        for condition_name in part:
            condition = part[condition_name]
            print(f'found {len(condition.keys())} images in {condition.name}')
            instance_list = []
            for image_name in condition:
                image = condition[image_name]
                for instance_name in image:
                    if instance_name in ['height', 'width', 'K', 'depth', 'color']: continue
                    full_name = image[instance_name].name
                    instance_list.append(full_name)

        # for each kind of part
        if mode.startswith('train_test'):
            n_use = int(use_ratio * len(instance_list))
            instances = random.sample(instance_list, n_use)
            train, test = train_test_split(instances, test_size=test_ratio)
            with open(os.path.join(out_path, 'test.txt'), 'w') as f:
                f.write('\n'.join(test))
            with open(os.path.join(out_path, 'train.txt'), 'w') as f:
                f.write('\n'.join(train))

        elif mode.startswith('eval'):
            n_use = int(use_ratio * len(instance_list))
            print(f'use ratio is {use_ratio}, so choose {n_use} instances for eval from totally {len(instance_list)} images.')
            instance_list = random.sample(instance_list, n_use)
            with open(os.path.join(out_path, f'{mode}.txt'), 'w') as f:
                f.write('\n'.join(instance_list))

h.close()