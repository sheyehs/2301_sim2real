import os
import random
import h5py

i_mode = 0
out_dir = '../split'

#############################################

mode_list = [
    'synthetic',
    'real',
]

mode = mode_list[i_mode]

"""
train mode splits train and test data used in train.py
eval mode chooses data for eval.py which visualizes the predicted position onto original image.  
"""

if mode == 'synthetic':
    use_ratio = 1  # by object type
    test_ratio = 0.1
    eval_ratio = 0.1
    data_path = '../data.hdf5'
    out_train_name = 'train.txt'
    out_test_name = 'test.txt'
    out_eval_name = 'eval.txt'
    from sklearn.model_selection import train_test_split
elif mode == 'real':
    use_ratio = 1
    data_path = '../data_real.hdf5'

##################################################################

h = h5py.File(data_path, 'r')

for company_name in h:
    company = h[company_name]
    for part_name in company:
        # collect instances. depend on data storage structure
        part = company[part_name]
        out_path = os.path.join(out_dir, part_name)
        os.makedirs(out_path, exist_ok=True)
        instance_list = []  # for each kind of part
        for condition_name in part:
            condition = part[condition_name]
            print(f'Found {len(condition.keys())} images in {condition.name}.')
            for image_name in condition:
                image = condition[image_name]
                for instance_name in image:
                    if instance_name in ['height', 'width', 'K', 'depth', 'color']: continue
                    full_path = image[instance_name].name
                    instance_list.append(full_path)

        # for each type of part
        if mode.startswith('synthetic'):
            if use_ratio != 1:
                n_use = int(use_ratio * len(instance_list))
                instance_list = random.sample(instance_list, n_use)
            train, test = train_test_split(instance_list, test_size=test_ratio+eval_ratio)
            test, eval = train_test_split(test, test_size=eval_ratio/(test_ratio+eval_ratio))
            with open(os.path.join(out_path, out_train_name), 'w') as f:
                f.write('\n'.join(train))
            with open(os.path.join(out_path, out_test_name), 'w') as f:
                f.write('\n'.join(test))
            with open(os.path.join(out_path, out_eval_name), 'w') as f:
                f.write('\n'.join(eval))
            print(f'Choose {len(train)} train, {len(test)} test, {len(eval)} eval data for {part_name}.')

        elif mode.startswith('real'):
            n_use = int(use_ratio * len(instance_list))
            print(f'use ratio is {use_ratio}, so choose {n_use} instances for eval from totally {len(instance_list)} images.')
            instance_list = random.sample(instance_list, n_use)
            with open(os.path.join(out_path, f'{mode}.txt'), 'w') as f:
                f.write('\n'.join(instance_list))

h.close()