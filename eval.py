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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
parser.add_argument('--model', type=str, default='./results/0208_lr_0.0001/pose_model_99_0.038202.pth',  help='Evaluation model')
parser.add_argument('--dataset', type=str, default='data.hdf5', help='dataset root dir')
parser.add_argument('--split_dir', type=str, default='./split')
parser.add_argument('--split_file', type=str, default='eval.txt')
parser.add_argument('--pcd_dir', type=str, default='./models')
parser.add_argument('--num_objects', type=int, default=10)
args = parser.parse_args()

date = time.strftime("%m%d_%H%M")
output_result_dir = f'./results_eval/{date}'
os.makedirs(output_result_dir, exist_ok=True)
out_image_dir = f'./results_eval/{date}/images'
os.makedirs(out_image_dir, exist_ok=True)

image_shape = (400, 640, 3)
num_points = 500
num_rotations = 60
bs = 1
knn = KNearestNeighbor(1)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
estimator = PoseNet(num_points=num_points, num_obj=args.num_objects, num_rot=num_rotations)
estimator.cuda()
estimator.load_state_dict(torch.load(args.model))
estimator.eval()

def main():
    test_dataset = PoseDataset('eval', num_points, False, args.dataset, 0.0, args.split_dir)  # add args.split_file
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)
    sym_list = test_dataset.get_sym_list()
    # rot_anchors = torch.from_numpy(estimator.rot_anchors).float().cuda()
    diameter = test_dataset.get_diameter()
    part_list = test_dataset.get_part_list()

    success_count = [0 for _ in range(args.num_objects)]
    num_count = [0 for _ in range(args.num_objects)]
    fw = open(f'{output_result_dir}/eval_result_logs.txt', 'w')

    error_data = 0
    for i, data in enumerate(test_dataloader, 0):
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
        model_points = model_points[0].cpu().detach().numpy()
        pred = np.dot(model_points, pred_r.T) + pred_t
        target = target_r[0].cpu().detach().numpy() + gt_t.cpu().data.numpy()[0]
        save_images(idx[0].item(), instance_path, K, color, pred * 1000, target * 1000)

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
            print('{0} Pass! Distance: {1}'.format(instance_path, dis))
            fw.write('{0} Pass! Distance: {1}\n'.format(instance_path, dis))
        else:
            print('{0} NOT Pass! Distance: {1}'.format(instance_path, dis))
            fw.write('{0} NOT Pass! Distance: {1}\n'.format(instance_path, dis))
        num_count[idx[0].item()] += 1

    accuracy = 0.0
    for i in range(args.num_objects):
        accuracy += float(success_count[i]) / num_count[i]
        print('Object {0} success rate: {1}'.format(part_list[i], float(success_count[i]) / num_count[i]))
        fw.write('Object {0} success rate: {1}\n'.format(part_list[i], float(success_count[i]) / num_count[i]))
    print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
    print('Accuracy: {0}'.format(accuracy / args.num_objects))
    fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
    fw.write('Accuracy: {0}\n'.format(accuracy / args.num_objects))
    fw.write('{0} corrupted data'.format(error_data))
    fw.close()

def save_images(part_idx, instance_path, K, color, pred, target):
    K = K[0].numpy()
    instance_path = instance_path[0].lstrip('/')
    color = color[0].numpy()

    # part_name = part_list[part_idx]
    # pcd_path = f'{pcd_dir}/{part_name}.master.ply'
    # pcd = o3d.io.read_point_cloud(pcd_path)
    # pcd = np.array(pcd.points)

    color = projection(K, pred, 'red', color)
    color = projection(K, target, 'green', color)

    file_path = os.path.join(out_image_dir, instance_path)
    os.makedirs(file_path[:file_path.rfind('/')], exist_ok=True)
    image = Image.fromarray(color)
    image.save(file_path + '.png')

def projection(K, points, hue, color):
    # points = np.matmul(pred_R, pcd.transpose()) + pred_t.transpose() # transform model points
    points = points.transpose()
    normalized_points = points / points[2]  # to normalized plane
    pixels = np.floor(np.matmul(K, normalized_points)[:2]).astype(int)  # project model points to image
    pixels[[0,1]] = pixels[[1,0]]  # swap (x ,y) to (y, x)
    keep = (pixels[0]>=0) & (pixels[0]<image_shape[0]) & (pixels[1]>=0) & (pixels[1]<image_shape[1])
    pixels = pixels[:, keep]
    pixels = np.unique(pixels, axis=1)

    if hue == 'red':
        i_hue = np.full((1, pixels.shape[1]), 0, dtype=int)
    elif hue == 'green':
        i_hue = np.full((1, pixels.shape[1]), 1, dtype=int)

    pixels = np.concatenate([pixels, i_hue], axis=0)

    flat_color = color.ravel()
    flat_index_array = np.ravel_multi_index(pixels, image_shape)
    flat_color[flat_index_array] = 255
    color = flat_color.reshape(image_shape)

    return color


if __name__ == '__main__':
    main()

