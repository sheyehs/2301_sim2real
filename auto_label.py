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

def get_bbox(bbox, h, w):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    h = 1024
    w = 1280
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 840)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > h:
        delt = rmax - h
        rmax = h
        rmin -= delt
    if cmax > w:
        delt = cmax - w
        cmax = w
        cmin -= delt
    return rmin, rmax, cmin, cmax

def label_individual_pose(estimator, scene_img, scene_depth, obj_mask, intrinsics, cad_model_pcs, cad_model_faces, visualize=True):
    norm = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                                std=[0.229, 0.229, 0.229])])
    h, w = obj_mask.shape

    xmap = np.array([[i for i in range(w)] for j in range(h)])
    ymap = np.array([[j for i in range(w)] for j in range(h)])

    bbox = obj_mask.flatten().nonzero()[0]
    if not len(bbox) > 0:
        return None, None, None
    ys = bbox // w
    xs = bbox - ys * w
    bbox = [np.min(ys), np.min(xs), np.max(ys), np.max(xs)]
    rmin, rmax, cmin, cmax = get_bbox(bbox, h, w)

    choose = obj_mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    model_pc = uniform_sample(cad_model_pcs, cad_model_faces, 1000)

    predict_sRT = None
    num_points = 1000
    depth_scale = 1000.0
    if len(choose) > num_points / 10:
        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
        
        depth_masked = scene_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked / depth_scale
        pt0 = (xmap_masked - intrinsics[0, 2]) * pt2 / intrinsics[0, 0]
        pt1 = (ymap_masked - intrinsics[1, 2]) * pt2 / intrinsics[1, 1]
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        # resize cropped image to standard size and adjust 'choose' accordingly
        img_size = 192
        img_masked = copy.deepcopy(scene_img[rmin:rmax, cmin:cmax, :])
        img_masked = cv2.resize(img_masked, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        crop_w = rmax - rmin
        ratio = img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * img_size + np.floor(col_idx * ratio)).astype(np.int64)
        choose = np.array([choose])

        img_masked = norm(img_masked)
        cloud = cloud.astype(np.float32)

        cloud = torch.from_numpy(cloud.astype(np.float32))
        choose = torch.LongTensor(choose.astype(np.int32))
        index = torch.LongTensor([0])

        cloud = Variable(cloud).cuda()
        choose = Variable(choose).cuda()
        img_masked = Variable(img_masked).cuda()
        index = Variable(index).cuda()

        cloud = cloud.view(1, num_points, 3)
        img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

        pred_r, pred_t, pred_c = estimator(img_masked, cloud, choose, index)
        try:
            pred_t, pred_mask = ransac_voting_layer(cloud, pred_t)
        except RuntimeError:
            print('RANSAC voting fails')
            return predict_sRT, None, np.array(model_pc.points)[:, :3]
        
        my_t = pred_t.cpu().data.numpy()
        how_min, which_min = torch.min(pred_c, 1)
        my_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
        points = cloud.view(num_points, 3)

        predict_sRT = quaternion_matrix(my_r)
        predict_sRT[:3, 3] = my_t
    
    if predict_sRT is None:
        return predict_sRT, None, model_pc[:, :3]
    else:
        return predict_sRT, points.squeeze().detach().cpu().numpy(), model_pc[:, :3]  

# pts_in_camera: (1000, 3)
# pts_in_obj: (1000, 3)
# init_pose: (4, 4)
def measure_pseudo_pose_in_3d(pts_in_camera, pts_in_obj, init_pose, knn):
    transformed_obj_pts = transform_coordinates_3d(pts_in_obj.T, init_pose)
    transformed_camera_pts = pts_in_camera.T

    target_pts = torch.from_numpy(transformed_camera_pts).cuda()
    source_pts = torch.from_numpy(transformed_obj_pts).cuda()

    index1 = knn(target_pts.unsqueeze(0), source_pts.unsqueeze(0))
    index2 = knn(source_pts.unsqueeze(0), target_pts.unsqueeze(0))

    pts1 = torch.index_select(target_pts, 1, index1.view(-1) - 1)
    cd1 = torch.mean(torch.norm((source_pts.float() - pts1.float()), dim=0)).cpu().item()

    pts2 = torch.index_select(source_pts, 1, index2.view(-1) - 1)
    cd2 = torch.mean(torch.norm((target_pts.float() - pts2.float()), dim=0)).cpu().item()
    return cd1 + cd2

def measuring_mask_in_2d(pseudo_mask, observed_mask):
    positive_pixels = pseudo_mask[observed_mask > 0]
    negative_pixels = observed_mask[pseudo_mask == 0]

    positive_error = sum(abs(1 - positive_pixels)) / (len(positive_pixels) + 1)
    negative_error = sum(negative_pixels) / (len(negative_pixels) + 1)

    return positive_error + negative_error

def measuring_rgb_in_2d(pseudo_rgb, observed_rgb, observed_mask):
    return measure_perceptual_similarity(pseudo_rgb, observed_rgb, observed_mask)

def measuring_pseudo_pose_in_2d(obj_id, inst_id, pose, observed_img, observed_mask, ren, ren_model, render_height, render_width, intrinsics):
    horizontal_indicies = np.where(np.any(observed_mask, axis=0))[0]
    vertical_indicies = np.where(np.any(observed_mask, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    x2 += 1
    y2 += 1

    img_height = observed_img.shape[0]
    img_width = observed_img.shape[1]

    rgb_img, _, mask_img, _ = croped_rendering(obj_id, pose, intrinsics, img_height, img_width, int(float(x1+x2) / 2), int(float(y1+y2) / 2), render_height, render_width, ren, ren_model)

    rgb_img = copy.deepcopy(rgb_img[y1:y2, x1:x2, :])
    observed_img = copy.deepcopy(observed_img[y1:y2, x1:x2, :])
    mask_img = copy.deepcopy(mask_img[y1:y2, x1:x2])
    observed_mask = copy.deepcopy(observed_mask[y1:y2, x1:x2])

    mask_error = measuring_mask_in_2d(mask_img, observed_mask)
    perceptual_error = measuring_rgb_in_2d(rgb_img, observed_img, observed_mask)
    
    return mask_error, perceptual_error

def label_poses_with_teacher(iter_idx, root_dir):
    knn = KNearestNeighbor(1)

    # load the trained network for pose labelling
    estimator = PoseNet(num_points=1000, num_obj=10, num_rot=60)
    estimator.cuda()
    if iter_idx == 0:
        prev_model_path = os.path.join(root_dir, 'initial_model')
    else:
        prev_model_path = os.path.join(root_dir, f'iteration_{iter_idx-1:02}')
    estimator.load_state_dict(torch.load(prev_model_path))
    estimator.eval()
    
    # should create another mode for train?
    dataset = PoseDataset('eval', num_points, False, args.dataset, 0.0, args.split_dir, args.split_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)
    sym_list = dataset.get_sym_list()
    diameters = dataset.get_diameter()
    part_list = dataset.get_part_list()

    # trverse all scenes for labelling
    scene_poses = []
    scene_residuals = [] # 3d error
    scene_mask_errors = []
    scene_rgb_errors = []
    scene_2d_errors = []
    scene_valid_iid = []

    instances = []

    total_inst_num = 0

    for i, data in enumerate(dataloader):
        points, choose, img, target_r, model_points, idx, gt_t, instance_path, K, color = data
        obj_diameter = diameters[idx.item()]
        points, choose, img, target_r, model_points, idx = Variable(points).cuda(), \
                                                            Variable(choose).cuda(), \
                                                            Variable(img).cuda(), \
                                                            Variable(target_r).cuda(), \
                                                            Variable(model_points).cuda(), \
                                                            Variable(idx).cuda()


        # sample points
        observed_mask = np.equal(mask, inst_id)
        current_mask = np.equal(mask, inst_id)
        current_mask = np.logical_and(current_mask, depth > 0)  # valid depth

        pred_r, pred_t, pred_c = estimator(img, points, choose, index)
        try:
            pred_t, pred_mask = ransac_voting_layer(points, pred_t)
        except RuntimeError:
            print('RANSAC voting fails')
            continue
        
        pred_t = pred_t.cpu().data.numpy()
        how_min, which_min = torch.min(pred_c, 1)
        pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
        points = points.view(num_points, 3)
        pred_r = quaternion_matrix(pred_r)[:3, :3]

        pts_in_obj = np.matmul(model_points, pred_r.T) + pred_t
    
        # compute error
        dis_3d = measure_pseudo_pose_in_3d(points, pts_in_obj, pred_r, pred_t, knn)
        
        mask_error, _ = measuring_pseudo_pose_in_2d(obj_id, total_inst_num, pose_from_teacher, img, observed_mask, ren, ren_model, render_height, render_width, intrinsics)

        scene_poses.append((pred_r, pred_t))

        scene_residuals.append(dis_3d)
        scene_mask_errors.append(mask_error)
        scene_rgb_errors.append(rgb_error)
        scene_2d_errors.append(mask_error)
        instances.append(instance_path)

    # compute statistics information for all residuals
    mean_residual = np.mean(np.array(scene_residuals))
    var_residual = np.std(np.array(scene_residuals))
    threshold_3d_error = mean_residual + var_residual

    mean_2d_error = np.mean(np.array(scene_2d_errors))
    var_2d_error = np.std(np.array(scene_2d_errors))
    threshold_2d_error = mean_2d_error + var_2d_error

    # filtering out the pose whose residual is larger than the threshold
    scene_good_poses = []
    good_scene_names = []

    valid_inst_num = 0
    for i in range(len(scene_poses)):

        if scene_2d_errors[i] < threshold_2d_error and scene_residuals[i] < threshold_3d_error:
            good_scene_names.append(instances[i])
            scene_good_poses.append(scene_poses[i])
            valid_inst_num += 1

    # also need to update the label.pkl and the mask.png at the same time
    teacher_label_dir = os.path.join(root_dir, f'iteration_{iter_idx:02}')
    os.makedirs(teacher_label_dir, exist_ok=True)
    
"""
teacher_labels_R_t.hdf5 hierarchy
company | part  | condition | image | instance  | R
                                                | t

"""
    h_path = os.path.join(teacher_label_dir, 'teacher_labels_R_t.hdf5')
    h = h5py.File(h_path,'a')

    for i in range(len(good_scene_names)):
        instance_grp = h.require_group(good_scene_names[i])
        h.create_dataset('R', data=np.array(scene_good_poses[i][0]))
        h.create_dataset('t', data=np.array(scene_good_poses[i][1]))

    del knn

if __name__ == '__main__':
    pass
