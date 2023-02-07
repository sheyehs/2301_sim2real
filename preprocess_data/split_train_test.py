import os
import random
import h5py

mode = 'train'
data_path = '../data.hdf5'
out_dir = '../split_new/'

#################################################################

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