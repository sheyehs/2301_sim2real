import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from dataset import PoseDataset
from lib.network import PoseNet
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.transformations import quaternion_matrix
from lib.knn.__init__ import KNearestNeighbor
import open3d as o3d
from PIL import Image
import time

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', type=str, default='./split')
    parser.add_argument('--part', type=str, help='part name')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--noise_trans', default=0.0, help='random noise added to translation')
    parser.add_argument('--model', type=str, default='',  help='Evaluation model')
    parser.add_argument('--pcd_dir', type=str, default='./models')
    return parser.parse_args()

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    opt.dataset_path = './data_real.hdf5'
    opt.num_objects = 1
    opt.result_dir = f'./results_eval/{time.strftime("%m%d_%H%M")}_{opt.part}'
    os.makedirs(opt.result_dir, exist_ok=True)
    opt.result_image_dir = f'{opt.result_dir}/images'
    os.makedirs(opt.result_image_dir, exist_ok=True)

    opt.image_shape = (400, 640, 3)
    opt.num_rot = 60
    opt.num_depth_pixels = 500
    opt.num_mesh_points = 500
    opt.split_file = 'eval_on_real.txt'
    knn = KNearestNeighbor(1)

    estimator = PoseNet(num_points=opt.num_depth_pixels, num_obj=opt.num_objects, num_rot=opt.num_rot)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()

    test_dataset = PoseDataset('eval', opt, opt.split_file, False, opt.noise_trans)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)
    
    sym_list = test_dataset.get_sym_list()
    # rot_anchors = torch.from_numpy(estimator.rot_anchors).float().cuda()
    diameter = test_dataset.get_diameter()
    idx = torch.tensor([0], dtype=int).cuda()

    success_count = np.zeros(4, dtype=int)
    num_count = 0
    test_dis = 0.0
    fw = open(f'{opt.result_dir}/eval_result_logs.txt', 'w')
    error_data = 0

    for i, data in enumerate(test_dataloader, 0):
        points, choose, img, target_r, model_points, gt_t, instance_path, K, color = data
        points, choose, img, target_r, model_points = Variable(points).cuda(), \
                                                                    Variable(choose).cuda(), \
                                                                    Variable(img).cuda(), \
                                                                    Variable(target_r).cuda(), \
                                                                    Variable(model_points).cuda()
        with torch.no_grad():
            pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
        how_min, which_min = torch.min(pred_c, 1)
        pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
        pred_r = quaternion_matrix(pred_r)[:3, :3]
        try:
            pred_t, pred_mask = ransac_voting_layer(points, pred_t)
        except:
            print("RANSAC failed. Skipped.")
            error_data += 1
            continue
        pred_t = pred_t.cpu().data.numpy()
        model_points = model_points[0].cpu().detach().numpy()
        pred = np.dot(model_points, pred_r.T) + pred_t
        target = target_r[0].cpu().detach().numpy() + gt_t[0].cpu().data.numpy()
        save_images(instance_path, K, color, pred * 1000, target * 1000)

        if idx[0].item() in sym_list:
            pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
            target = torch.index_select(target, 1, inds.view(-1) - 1)
            dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
        else:
            dis = np.mean(np.linalg.norm(pred - target, axis=1))

        error_ratio = dis / diameter
        cond = [error_ratio < 0.05, error_ratio < 0.1, error_ratio < 0.2, error_ratio < 0.5] 
        success_count[cond] += 1
        num_count += 1
        test_dis += dis
        print('{0} Distance: {1}'.format(instance_path, dis))
        fw.write('{0} Distance: {1}\n'.format(instance_path, dis))

    accuracy = success_count / num_count
    test_dis = test_dis / num_count 
    
    fw.write(f'accuracy of 0.05 diameter: {accuracy[0]}\n')
    fw.write(f'accuracy of 0.1 diameter: {accuracy[1]}\n')
    fw.write(f'accuracy of 0.2 diameter: {accuracy[2]}\n')
    fw.write(f'accuracy of 0.5 diameter: {accuracy[3]}\n')
    fw.write(f'average distance error: {test_dis}\n')
    fw.write(f'average t error:\n')  # todo
    fw.write(f'average R error:\n')  # todo
    fw.write('{0} corrupted data\n'.format(error_data))
    fw.close()

def save_images(instance_path, K, color, pred, target):
    K = K[0].numpy()
    instance_path = instance_path[0].lstrip('/')
    color = color[0].numpy()

    # part_name = part_list[part_idx]
    # pcd_path = f'{pcd_dir}/{part_name}.master.ply'
    # pcd = o3d.io.read_point_cloud(pcd_path)
    # pcd = np.array(pcd.points)

    color = projection(K, pred, 'red', color)
    color = projection(K, target, 'green', color)

    file_path = os.path.join(opt.result_image_dir, instance_path)
    os.makedirs(file_path[:file_path.rfind('/')], exist_ok=True)
    image = Image.fromarray(color)
    image.save(file_path + '.png')

def projection(K, points, hue, color):
    # points = np.matmul(pred_R, pcd.transpose()) + pred_t.transpose() # transform model points
    points = points.transpose()
    normalized_points = points / points[2]  # to normalized plane
    pixels = np.floor(np.matmul(K, normalized_points)[:2]).astype(int)  # project model points to image
    pixels[[0,1]] = pixels[[1,0]]  # swap (x ,y) to (y, x)
    keep = (pixels[0]>=0) & (pixels[0]<opt.image_shape[0]) & (pixels[1]>=0) & (pixels[1]<opt.image_shape[1])
    pixels = pixels[:, keep]
    pixels = np.unique(pixels, axis=1)

    if hue == 'red':
        i_hue = np.full((1, pixels.shape[1]), 0, dtype=int)
    elif hue == 'green':
        i_hue = np.full((1, pixels.shape[1]), 1, dtype=int)

    pixels = np.concatenate([pixels, i_hue], axis=0)

    flat_color = color.ravel()
    flat_index_array = np.ravel_multi_index(pixels, opt.image_shape)
    flat_color[flat_index_array] = 255
    color = flat_color.reshape(opt.image_shape)

    return color


if __name__ == '__main__':
    opt = options()
    main(opt)

