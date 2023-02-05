import os
import random
import h5py

mode = 'eval'
data_path = '../data_real.hdf5'
out_path = '../split/'

if mode == 'train':
    use_ratio = 0.025
    test_ratio = 0.1
    from sklearn.model_selection import train_test_split
elif mode == 'eval':
    pass

h = h5py.File(data_path, 'r')

for company_name in h:
    company = h[company_name]
    for part_name in company:
        out_path = os.path.join(out_root, part_name)
        os.makedirs(out_path, exist_ok=True)
        part = company[part_name]
        for condition_name in part:
            instance_list = []
            condition = part[condition_name]
            print(f'found {len(condition.keys())} images in {condition.name}')
            for image_name in condition:
                image = condition[image_name]
                for instance_name in image:
                    if instance_name in ['height', 'width', 'K', 'depth', 'color']: continue
                    name = condition.name + '/' + image_name + '/' + instance_name
                    instance_list.append(name)

            if mode == 'train':
                n_use = int(use_ratio * len(instance_list))
                instances = random.sample(instance_list, n_use)
                train, test = train_test_split(instances, test_size=test_ratio)
                with open(os.path.join(out_path, 'test.txt'), 'w') as f:
                    f.write('\n'.join(test))
                with open(os.path.join(out_path, 'train.txt'), 'w') as f:
                    f.write('\n'.join(train))

            elif mode == 'eval':
                with open(os.path.join(out_path, 'eval.txt'), 'w') as f:
                    f.write('\n'.join(instance_list))

h.close()