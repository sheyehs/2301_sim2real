import glob
import cv2
import h5py
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.transformations import quaternion_matrix
from lib.network import PoseNet
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.utils import load_obj, uniform_sample, transform_coordinates_3d
from lib.knn.__init__ import KNearestNeighbor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from renderer.rendering import *
from perceptual_loss.alexnet_perception import *

# pts_in_camera: (1000, 3)
# pts_in_obj: (1000, 3)
# init_pose: (4, 4)
def measure_pseudo_pose_in_3d(pts_in_camera, pts_in_obj, R, t, knn):
    source_pts = np.matmul(R, pts_in_obj.T) + t.T
    target_pts = pts_in_camera.T

    # target_pts = torch.from_numpy(transformed_camera_pts).cuda()
    # source_pts = torch.from_numpy(transformed_obj_pts).cuda()

    index1 = knn(target_pts.unsqueeze(0), source_pts.unsqueeze(0))  # is the order right?
    index2 = knn(source_pts.unsqueeze(0), target_pts.unsqueeze(0))

    pts1 = torch.index_select(target_pts, 1, index1.view(-1) - 1)
    cd1 = torch.mean(torch.norm((source_pts.float() - pts1.float()), dim=0)).cpu().item()

    pts2 = torch.index_select(source_pts, 1, index2.view(-1) - 1)
    cd2 = torch.mean(torch.norm((target_pts.float() - pts2.float()), dim=0)).cpu().item()
    return cd1 + cd2

def measuring_mask_in_2d(mask, pcd, bbox, pred_R, pred_t, K):
    # project pcd to get pseudo_mask
    points = np.matmul(pred_R, pcd.T) + pred_t
    pixels = np.matmul(K, points / points[2]).astype(int)[:2]
    pixels = np.unique(pixels, axis=1)
    pixels[[0,1]] = pixels[[1,0]]

    y_min, y_max, x_min, x_max = bbox
    pixels = pixels[:, pixels[0]>=y_min & pixels[0]<y_max & pixels[1]>=x_min & pixels[1]<x_max]

    pseudo_mask = np.full(image_shape, False, dtype=bool)
    pseudo_mask.take(pixels) = True
    pseudo_mask = pseudo_mask[y_min:y_max, x_min:x_max]

    positive_pixels = pseudo_mask[observed_mask > 0]
    negative_pixels = observed_mask[pseudo_mask == 0]

    positive_error = sum(abs(1 - positive_pixels)) / (len(positive_pixels) + 1)
    negative_error = sum(negative_pixels) / (len(negative_pixels) + 1)

    return positive_error + negative_error


def label_poses_with_teacher(iter_idx, root_dir, args):
    knn = KNearestNeighbor(1)

    # load the trained network for pose labelling
    estimator = PoseNet(num_points=500, num_obj=10, num_rot=60)
    estimator.cuda()
    if iter_idx == 0:
        prev_model_path = os.path.join(root_dir, 'initial', 'model.pth')
    else:
        prev_model_path = os.path.join(root_dir, f'iteration_{iter_idx-1:02}', 'model.pth')
    estimator.load_state_dict(torch.load(prev_model_path))
    estimator.eval()
    
    dataset = PoseDataset('self_train', num_points, False, args.dataset, 0.0, args.split_dir, args.split_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)
    sym_list = dataset.get_sym_list()
    diameters = dataset.get_diameter()
    part_list = dataset.get_part_list()

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
        points, choose, img, target_r, model_points, idx, gt_t, instance_path, K, color = data
        obj_diameter = diameters[idx.item()]  # will be used to normalized error
        points, choose, img, target_r, model_points, idx = Variable(points).cuda(), \
                                                            Variable(choose).cuda(), \
                                                            Variable(img).cuda(), \
                                                            Variable(target_r).cuda(), \
                                                            Variable(model_points).cuda(), \
                                                            Variable(idx).cuda()

        # need pass the original mask and valid depth mask

        # observed_mask = np.equal(mask, inst_id)
        # current_mask = np.equal(mask, inst_id)
        # current_mask = np.logical_and(current_mask, depth > 0)

        pred_r, pred_t, pred_c = estimator(img, points, choose, index)

        try:
            pred_t, pred_mask = ransac_voting_layer(points, pred_t)
        except RuntimeError:
            print('RANSAC voting fails')
            continue
        
        pred_t = pred_t.cpu().data.numpy()
        how_min, which_min = torch.min(pred_c, 1)
        pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
        pred_r = quaternion_matrix(pred_r)[:3, :3]
        pts_in_obj = np.matmul(model_points, pred_r.T) + pred_t

        pts_in_camera = points.view(num_points, 3)  # pts in camera
    
        # compute error
        dis_3d = measure_pseudo_pose_in_3d(points, pts_in_obj, pred_r, pred_t, knn)
        dis_3d /= obj_diameter
        
        mask_error = measuring_mask_in_2d(mask, pcd, bbox, pred_R, pred_t, K)
        error_2d = mask_error

        pred_poses.append((pred_r, pred_t))
        errors_3d.append(dis_3d)
        # mask_errors.append(mask_error)
        # rgb_errors.append(rgb_error)
        errors_2d.apend(error_2d)
        instance_paths.append(instance_path)

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
    teacher_label_dir = os.path.join(root_dir, f'iteration_{iter_idx:02}')
    os.makedirs(teacher_label_dir, exist_ok=True)

    h_path = os.path.join(teacher_label_dir, 'teacher_labels.hdf5')
    h = h5py.File(h_path,'a')

    good_instance_paths = []
    for i in range(len(pred_poses)):
        if errors_2d[i] < threshold_2d_error and errors_3d[i] < threshold_3d_error:
            good_instance_path = instance_paths[i]
            good_instance_paths.append(good_instance_path)

            instance_grp = h.require_group(good_instance_path)
            h.create_dataset('R', data=np.array(pred_poses[i][0]))
            h.create_dataset('t', data=np.array(pred_poses[i][1]))

    with open(os.path.join(teacher_label_dir, 'good_instances.txt'), 'w') as f:
        f.write('\n'.join(good_instance_paths))

    print(f'Iteration {iter_idx}: After filtering, {len(good_instance_paths)}/{len(pred_poses)} good labels are created \
            for student learning. Good rate is {len(good_instance_paths)/len(pred_poses)}.')

    h.close()
    del knn

if __name__ == '__main__':
    pass
