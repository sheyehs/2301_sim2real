import os
import glob
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from lib.transformations import quaternion_matrix
from lib.network import PoseNet
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.knn.__init__ import KNearestNeighbor
from dataset import PoseDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def measure_pseudo_pose_in_3d(pts_in_camera, pts_in_obj, knn):
    source_pts = pts_in_obj.squeeze(0).T  # (3, 500)
    target_pts = pts_in_camera.T  # (3, 500)

    index1 = knn.apply(target_pts.unsqueeze(0), source_pts.unsqueeze(0), 1)  # is the order right?
    index2 = knn.apply(source_pts.unsqueeze(0), target_pts.unsqueeze(0), 1)

    pts1 = torch.index_select(target_pts, 1, index1.view(-1) - 1)
    cd1 = torch.mean(torch.norm((source_pts.float() - pts1.float()), dim=0)).cpu().item()

    pts2 = torch.index_select(source_pts, 1, index2.view(-1) - 1)
    cd2 = torch.mean(torch.norm((target_pts.float() - pts2.float()), dim=0)).cpu().item()

    return cd1 + cd2

def measuring_mask_in_2d(observed_mask, pcd, bbox, pred_R, pred_t, K, image_shape):
    # project pcd to get pseudo_mask
    pcd = torch.from_numpy(pcd).float().cuda()
    # print(pcd.shape)
    points = torch.matmul(pred_R, pcd.T) + pred_t.T
    # print(points.shape)
    pixels = torch.floor(torch.matmul(K, points / points[2])[:2]).int()
    # print(K.shape)
    # print(pixels.shape)
    pixels = torch.unique(pixels, dim=1)
    # print(pixels.shape)
    pixels[[0,1]] = pixels[[1,0]]
    # print(pixels.shape)

    y_min, y_max, x_min, x_max = bbox
    # print(bbox)
    pixels = pixels[:, (pixels[0]>=y_min) & (pixels[0]<y_max) & (pixels[1]>=x_min) & (pixels[1]<x_max)]
    # print(pixels.shape)
    flat_pseudo_mask = np.full(image_shape, False, dtype=bool).ravel()
    flat_index = np.ravel_multi_index(pixels.cpu().data.numpy(), image_shape)
    flat_pseudo_mask[flat_index] = True
    pseudo_mask = torch.from_numpy(flat_pseudo_mask.reshape(image_shape))

    pseudo_mask = pseudo_mask[y_min:y_max, x_min:x_max]

    # print(pseudo_mask.shape, observed_mask.shape)
    positive_pixels = pseudo_mask[observed_mask > 0].int()
    negative_pixels = observed_mask[pseudo_mask == 0].int()

    positive_error = sum(abs(1 - positive_pixels)) / (len(positive_pixels) + 1)
    negative_error = sum(negative_pixels) / (len(negative_pixels) + 1)

    return (positive_error + negative_error).cpu().item()


def label_poses_with_teacher(iter_idx, root_dir, args):
    knn = KNearestNeighbor(1)

    # load the trained network for pose labelling
    estimator = PoseNet(num_points=500, num_obj=10, num_rot=60)
    estimator.cuda()
    teacher_path = os.path.join(root_dir, 'best', 'best_model.pth')
    estimator.load_state_dict(torch.load(teacher_path))
    estimator.eval()
    
    dataset = PoseDataset('teacher', 500, False, args.dataset, 0.0, args.split_dir, args.split_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)
    sym_list = dataset.get_sym_list()
    diameters = dataset.get_diameter()
    part_list = dataset.get_part_list()
    pcd_list = dataset.get_pcd_list()

    # trverse all scenes for labelling
    pred_poses = []
    errors_3d = []
    # mask_errors = []
    # rgb_errors = []
    errors_2d = []
    instance_paths = []

    total_inst_num = 0

    for i, data in enumerate(dataloader):
        # add return mask, and full model pcd, bbox, image shape
        points, choose, img, model_points, idx, instance_path, K, image_shape, bbox, mask = data
        obj_diameter = diameters[idx.item()]  # will be used to normalized error
        pcd = pcd_list[part_list[idx.item()]]  # to project with pred R, t
        points, choose, img, model_points, idx, K, bbox, mask = Variable(points).cuda(), \
                                                            Variable(choose).cuda(), \
                                                            Variable(img).cuda(), \
                                                            Variable(model_points).cuda(), \
                                                            Variable(idx).cuda(), \
                                                            Variable(K).float().cuda()[0], \
                                                            Variable(bbox).int().cuda()[0], \
                                                            Variable(mask).float().cuda()[0]

        with torch.no_grad():
            pred_r, pred_t, pred_c = estimator(img, points, choose, idx)

        try:
            pred_t, pred_mask = ransac_voting_layer(points, pred_t)
        except RuntimeError:
            print('RANSAC voting fails')
            continue
        pred_t = pred_t.view(1, 3)

        how_min, which_min = torch.min(pred_c, 1)
        pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
        pred_r = torch.from_numpy(quaternion_matrix(pred_r)[:3, :3]).float().cuda()
        pts_in_obj = torch.matmul(model_points, pred_r.T) + pred_t  # (500, 3)

        pts_in_camera = points.view(500, 3)  # pts in camera
    
        # compute error
        dis_3d = measure_pseudo_pose_in_3d(pts_in_camera, pts_in_obj, knn)
        dis_3d /= obj_diameter
        
        mask_error = measuring_mask_in_2d(mask, pcd, bbox, pred_r, pred_t, K, image_shape)
        error_2d = mask_error

        pred_poses.append((pred_r.reshape(3, 3).cpu().data.numpy(), pred_t.reshape(3, 1).cpu().data.numpy()))
        errors_3d.append(dis_3d)
        # mask_errors.append(mask_error)
        # rgb_errors.append(rgb_error)
        errors_2d.append(error_2d)
        instance_paths.append(instance_path[0])

    del knn

    # compute statistics information for all residuals
    mean_3d_error = np.mean(np.array(errors_3d))
    std_3d_error = np.std(np.array(errors_3d))
    threshold_3d_error = mean_3d_error + std_3d_error

    mean_2d_error = np.mean(np.array(errors_2d))
    std_2d_error = np.std(np.array(errors_2d))
    threshold_2d_error = mean_2d_error + std_2d_error

    # filtering out the pose whose residual is larger than the threshold
    """
    teacher_labels.hdf5 hierarchy
    company | part  | condition | image | instance  | R
                                                    | t
    """
    iteration_dir = os.path.join(root_dir, f'iteration_{iter_idx:02}')
    os.makedirs(iteration_dir, exist_ok=True)

    h_path = os.path.join(iteration_dir, 'teacher_labels.hdf5')
    h = h5py.File(h_path,'a')

    good_instance_paths = {}
    for i in range(len(pred_poses)):
        if errors_2d[i] < threshold_2d_error and errors_3d[i] < threshold_3d_error:
            good_instance_path = instance_paths[i]
            part_name = good_instance_path.split('/')[-4]
            if part_name in good_instance_paths.keys():
                good_instance_paths[part_name].append(good_instance_path)
            else:
                good_instance_paths[part_name] = [good_instance_path]

            instance_grp = h.require_group(good_instance_path)
            instance_grp.create_dataset('R', data=np.array(pred_poses[i][0]))
            instance_grp.create_dataset('t', data=np.array(pred_poses[i][1]))
    h.close()

    test_ratio = 0.1
    for part_name, instances in good_instance_paths.items():
        print(f"Select {len(instances)} good instances for {part_name}")
        train, test = train_test_split(instances, test_size=test_ratio)
        os.makedirs(os.path.join(iteration_dir, 'good_instances', part_name), exist_ok=True)
        with open(os.path.join(iteration_dir, 'good_instances', part_name, 'train.txt'), 'w') as f:
            f.write('\n'.join(train))
        with open(os.path.join(iteration_dir, 'good_instances', part_name, 'test.txt'), 'w') as f:
            f.write('\n'.join(test))

    print(f'Iteration {iter_idx}: After filtering, {len(good_instance_paths)}/{len(pred_poses)} good labels are created \
            for student learning. Good rate is {len(good_instance_paths)/len(pred_poses)}.')

if __name__ == '__main__':
    pass
