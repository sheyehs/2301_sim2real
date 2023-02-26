from auto_label import label_poses_with_teacher
import os
import sys
import glob
import argparse
import time
import shutil
import numpy as np
import torch
from torch.autograd import Variable
from lib.network import PoseNet
from lib.loss import Loss
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.transformations import quaternion_matrix
from lib.knn.__init__ import KNearestNeighbor
from lib.utils import setup_logger
from dataset import PoseDataset
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--initial_model', type=str)
parser.add_argument('--renderer_model_dir', type=str, default='renderer/robi_models')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
parser.add_argument('--num_rot', type=int, default=60, help='number of rotation anchors')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--noise_trans', default=0.01, help='random noise added to translation')  # 5
parser.add_argument('--lr', default=0.00001, help='learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--resume_posenet', type=str, default='pose_model.pth', help='resume PoseNet model')
parser.add_argument('--nepoch', type=int, default=1, help='max number of epochs to train')  # 10
parser.add_argument('--iter_start', type=int, default=0, help='the start number of iterations for self-training')
parser.add_argument('--best_metric', type=float, default=100000.0)
parser.add_argument('--iter', type=int, default=50, help='the total number of iterations for self-training')
parser.add_argument('--validation', action='store_true', default=False, help='valid the real model with a small number of real data')
parser.add_argument('--dataset', type=str, default='./data_real.hdf5')
parser.add_argument('--split_dir', type=str, default='./split')
parser.add_argument('--split_file', type=str, default='eval_on_real.txt')

opt = parser.parse_args()

opt.num_objects = 10  # number of object classes in the dataset
opt.num_points = 500 # number of points selected from mask
opt.num_rot = 60


def self_training_with_real_data(iter_idx, root_dir, args, estimator_best_test=np.Inf):
    # set seed

    """
    root_dir    | best      | best_model.pth
                            | record.txt
                            | initial_model.pth
                | iteration | teacher_labels.hdf5
                            | good_instances    | part_name | train.txt
                                                            | test.txt
                            | iter_{iter_idx}_epoch_{epoch}_train.txt
                            | iter_{iter_idx}_epoch_{epoch}_test.txt
                            | pose_model_{epoch}_{dis}.pth
    """
    
    iteration_dir = os.path.join(root_dir, f'iteration_{iter_idx:02}')  # folder to save trained models
    os.makedirs(iteration_dir, exist_ok=True)

    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects, num_rot=opt.num_rot)
    estimator.cuda()
    teacher_path = os.path.join(root_dir, 'best', 'best_model.pth')
    estimator.load_state_dict(torch.load(teacher_path))

    label_root = os.path.join(iteration_dir, 'teacher_labels.hdf5')
    split_dir = os.path.join(iteration_dir, 'good_instances')
    
    dataset = PoseDataset('student', opt.num_points, True, opt.dataset, opt.noise_trans, split_dir, 'train.txt', label_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=10)
    test_dataset = PoseDataset('student', opt.num_points, False, opt.dataset, 0.0, split_dir, 'test.txt', label_root)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=10)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    opt.diameters = dataset.get_diameter()
    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<')
    print('length of the dataset: {0}'.format(len(dataset)))

    criterion = Loss(opt.sym_list, estimator.rot_anchors)
    knn = KNearestNeighbor(1)

    # add lr decay?

    optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.lr)
    global_step = 0
    st_time = time.time()
    best_test = estimator_best_test
    for epoch in range(opt.nepoch):
        logger = setup_logger(f'iter_{iter_idx}_epoch_{epoch}_train', os.path.join(iteration_dir, f'iter_{iter_idx}_epoch_{epoch}_train.txt'))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_loss_avg = 0.0
        train_loss_r_avg = 0.0
        train_loss_t_avg = 0.0
        train_loss_reg_avg = 0.0
        estimator.train()
        optimizer.zero_grad()
        for i, data in enumerate(dataloader):
            points, choose, img, target_t, target_r, model_points, idx, gt_t = data
            obj_diameter = opt.diameters[idx.item()]
            points, choose, img, target_t, target_r, model_points, idx = Variable(points).cuda(), \
                                                                            Variable(choose).cuda(), \
                                                                            Variable(img).cuda(), \
                                                                            Variable(target_t).cuda(), \
                                                                            Variable(target_r).cuda(), \
                                                                            Variable(model_points).cuda(), \
                                                                            Variable(idx).cuda()
            pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
            loss, loss_r, loss_t, loss_reg = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter)
            loss.backward()
            train_loss_avg += loss.item()
            train_loss_r_avg += loss_r.item()
            train_loss_t_avg += loss_t.item()
            train_loss_reg_avg += loss_reg.item()
            train_count += 1
            if train_count % opt.batch_size == 0:
                global_step += 1
                lr = opt.lr
                optimizer.step()
                optimizer.zero_grad()
                # add tensorboard writter
                logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_loss:{4:f}'.format(time.strftime("%Hh %Mm %Ss", 
                    time.gmtime(time.time()-st_time)), epoch, int(train_count/opt.batch_size), train_count, train_loss_avg/opt.batch_size))
                train_loss_avg = 0.0
                train_loss_r_avg = 0.0
                train_loss_t_avg = 0.0
                train_loss_reg_avg = 0.0

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger(f'iter_{iter_idx}_epoch_{epoch}_test', os.path.join(iteration_dir, f'iter_{iter_idx}_epoch_{epoch}_test.txt'))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'test started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        success_count = [0 for i in range(opt.num_objects)]
        num_count = [0 for i in range(opt.num_objects)]

        for j, data in enumerate(test_dataloader, 0):
            # target_t, target_r and gt_t are teacher labels
            points, choose, img, target_t, target_r, model_points, idx, gt_t = data
            obj_diameter = opt.diameters[idx.item()]
            points, choose, img, target_t, target_r, model_points, idx = Variable(points).cuda(), \
                                                                        Variable(choose).cuda(), \
                                                                        Variable(img).cuda(), \
                                                                        Variable(target_t).cuda(), \
                                                                        Variable(target_r).cuda(), \
                                                                        Variable(model_points).cuda(), \
                                                                        Variable(idx).cuda()
            with torch.no_grad():
                pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
            loss, _, _, _ = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter)
            test_count += 1
            how_min, which_min = torch.min(pred_c, 1)
            pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
            pred_r = quaternion_matrix(pred_r)[:3, :3]
            try:
                pred_t, pred_mask = ransac_voting_layer(points, pred_t)
            except RuntimeError:
                print('RANSAC voting fails')
                continue
            pred_t = pred_t.cpu().data.numpy()  # 1, 3
            model_points = model_points[0].cpu().detach().numpy()
            pred = np.dot(model_points, pred_r.T) + pred_t  # N, 3
            target = target_r[0].cpu().detach().numpy() + gt_t[0].cpu().data.numpy()
            if idx[0].item() in opt.sym_list:
                pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
                target = torch.index_select(target, 1, inds.view(-1) - 1)
                dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
            else:
                dis = np.mean(np.linalg.norm(pred - target, axis=1))
            logger.info('Test time {0} Test Frame No.{1} loss:{2:f} confidence:{3:f} distance:{4:f}'.format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss, how_min[0].item(), dis))
            if dis < 0.1 * opt.diameters[idx.item()]:
                success_count[idx[0].item()] += 1
            num_count[idx[0].item()] += 1
            test_dis += dis
        # compute accuracy
        accuracy = 0.0
        for i in range(opt.num_objects):
            accuracy += float(success_count[i]) / num_count[i]
            logger.info('Object {0} success rate: {1}'.format(test_dataset.objlist[i], float(success_count[i]) / num_count[i]))
        accuracy = accuracy / opt.num_objects
        test_dis = test_dis / test_count
        # log results
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2:f}, Accuracy: {3:f}'.format(time.strftime("%Hh %Mm %Ss",
            time.gmtime(time.time() - st_time)), epoch, test_dis, accuracy))      
        torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(iteration_dir, epoch, test_dis))  
        if test_dis < best_test:
            best_test = test_dis
            torch.save(estimator.state_dict(), '{0}/best/best_model.pth'.format(root_dir))
            with open('{0}/best/record.txt'.format(root_dir), 'a') as f:
                f.write(f'{time.strftime("%Hh %Mm %Ss", time.gmtime(time.time()-st_time))}: iter {iter_idx}, epoch {epoch}, test dist {test_dis}\n')
            logger.info('%d >>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<' % epoch)

        print('>>>>>>>>----------epoch {0} test finish---------<<<<<<<<'.format(epoch))
    
    return best_test

def iterative_self_training(start_iterations, num_iterations):
    estimator_best_test = opt.best_metric
    print('Self-training will start from {0}-th iteration and stop at {1}-th iteration.'.format(opt.iter_start, opt.iter-1))

    for iter in range(start_iterations, num_iterations):
        # if just transfer from virtual training to real training
        if iter == 0:
            if not os.path.exists(opt.initial_model):
                print('error, the initial model does not exist!')
                sys.exit()
            # prepare directory to store real models during self-training
            out_dir = f'./results_self-training/{time.strftime("%m%d_%H%M")}'
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'best'), exist_ok=True)
            # get trained virtual model
            shutil.copy(opt.initial_model, os.path.join(out_dir, 'best', 'best_model.pth'))
            shutil.copy(opt.initial_model, os.path.join(out_dir, 'best', 'initial_model.pth'))
            with open(os.path.join(out_dir, 'best', 'record.txt'), 'w') as f:
                f.write(f'Initial model: {opt.initial_model}\n')

            print(f'Initial model is: {opt.initial_model}')
            print(f'Self-training results will be output to {out_dir}')

        # # delete the old training data to save the storage
        # if iter >= 3:
        #     old_training_data_dir = os.path.join('./data', robi_object, 'teacher_label_iter_' + str(iter-1).zfill(2))
        #     shutil.rmtree(old_training_data_dir)

        # filter pseudo-labels from teacher
        label_poses_with_teacher(iter, out_dir, opt)
        # update numbers of real instances that have been used
        # update_train_test_split('./data/' + robi_object + '/teacher_label_iter_' + str(iter+1).zfill(2), './dataset/' + robi_object +'/dataset_config')
        # train student model
        estimator_best_test = self_training_with_real_data(iter, out_dir, opt, estimator_best_test)

if __name__ == '__main__':
    iterative_self_training(opt.iter_start, opt.iter)
