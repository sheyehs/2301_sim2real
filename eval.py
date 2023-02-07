import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from datasets.robotics.dataset import PoseDataset
from lib.network import PoseNet
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.transformations import quaternion_matrix
from lib.knn.__init__ import KNearestNeighbor
import open3d as o3d
from PIL import Image

out_image_dir = './eval_saved_images'
split_root = '/scratch/gc2720/2301_sim2real/split_new'
os.makedirs(out_image_dir, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
parser.add_argument('--model', type=str, default='/scratch/gc2720/2301_sim2real/results/lr_1e-4/pose_model_49_0.035039.pth',  help='Evaluation model')
parser.add_argument('--dataset_root', type=str, default='data.hdf5', help='dataset root dir')
opt = parser.parse_args()

part_list = [
    'SF-CJd60-097-016-016',
    'SF-CJd60-097-026',
    'SongFeng_0005',
    'SongFeng_306',
    'SongFeng_311',
    'SongFeng_318',
    'SongFeng_332',
    '21092302',
    '6010018CSV',
    '6010022CSV'
]
pcd_dir = './models'
image_shape = (400, 640, 3)

num_objects = 10
num_points = 500
num_rotations = 60
bs = 1
output_result_dir = 'eval_results'
if not os.path.exists(output_result_dir):
    os.makedirs(output_result_dir)
knn = KNearestNeighbor(1)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
estimator = PoseNet(num_points=num_points, num_obj=num_objects, num_rot=num_rotations)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

def save_projections(part_idx, instance_path, pred_R, pred_t, K, color):
    K = K[0].numpy()
    instance_path = instance_path[0].lstrip('/')
    color = color.numpy()[0]
    if color.shape != image_shape:
        print(instance_path)

    part_name = part_list[part_idx]
    pcd_path = f'{pcd_dir}/{part_name}.master.ply'
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = np.array(pcd.points)

    points = np.matmul(pred_R, pcd.transpose()) + pred_t.transpose() # transform model points
    normalized_points = points / points[2]  # to normalized plane
    pixels = np.floor(np.matmul(K, normalized_points)[:2]).astype(int)  # project model points to image
    pixels[[0,1]] = pixels[[1,0]]  # swap (x ,y) to (y, x)
    keep = (pixels[0]>=0) & (pixels[0]<image_shape[0]) & (pixels[1]>=0) & (pixels[1]<image_shape[1])
    pixels = pixels[:, keep]
    pixels = np.unique(pixels, axis=1)

    red = np.zeros((1, pixels.shape[1]), dtype=int)
    red_pixels = np.concatenate([pixels, red], axis=0)

    flat_color = color.ravel()
    flat_index_array = np.ravel_multi_index(red_pixels, image_shape)
    flat_color[flat_index_array] = 255
    color = flat_color.reshape(image_shape)

    file_path = os.path.join(out_image_dir, instance_path)
    os.makedirs(file_path[:file_path.rfind('/')], exist_ok=True)
    image = Image.fromarray(color)
    image.save(file_path + '.png')

test_dataset = PoseDataset('eval', num_points, False, opt.dataset_root, 0.0, split_root)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)
sym_list = test_dataset.get_sym_list()
rot_anchors = torch.from_numpy(estimator.rot_anchors).float().cuda()
diameter = test_dataset.get_diameter()

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

error_data = 0
for i, data in enumerate(test_dataloader, 0):
    # add return instance_path, K, color in eval mode
    points, choose, img, target_r, model_points, idx, gt_t, instance_path, K, color = data
    points, choose, img, target_r, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(target_r).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()
    pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
    pred_t, pred_mask = ransac_voting_layer(points, pred_t)
    pred_t = pred_t.cpu().data.numpy()
    how_min, which_min = torch.min(pred_c, 1)
    pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
    pred_r = quaternion_matrix(pred_r)[:3, :3]
    save_projections(idx[0].item(), instance_path, pred_r, pred_t, K, color)
    model_points = model_points[0].cpu().detach().numpy()
    pred = np.dot(model_points, pred_r.T) + pred_t
    target = target_r[0].cpu().detach().numpy() + gt_t.cpu().data.numpy()[0]

    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    if dis < 0.1 * diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        # print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1

accuracy = 0.0
for i in range(num_objects):
    accuracy += float(success_count[i]) / num_count[i]
    print('Object {0} success rate: {1}'.format(part_list[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(part_list[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
print('Accuracy: {0}'.format(accuracy / num_objects))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.write('Accuracy: {0}\n'.format(accuracy / num_objects))
fw.write('{0} corrupted data'.format(error_data))
fw.close()


