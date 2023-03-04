import os
import argparse
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from dataset import PoseDataset 
from lib.network import PoseNet
from lib.loss import Loss
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.transformations import quaternion_matrix
from lib.utils import setup_logger
from types import SimpleNamespace
import debugging

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only', type=bool, default=False)
    parser.add_argument('--eval_load_path', type=str)
    parser.add_argument('--seed', type=int)
    # data
    parser.add_argument('--part', type=str)
    parser.add_argument('--dataset_path', type=str, default='./data.hdf5') 
    parser.add_argument('--split_dir', type=str, default='./split')
    parser.add_argument('--split_train_file', type=str, default='train.txt')
    parser.add_argument('--split_test_file', type=str, default='test.txt')
    parser.add_argument('--split_eval_file', type=str, default='eval_on_real.txt')
    parser.add_argument('--output_dir', type=str, default='./results')
    # model
    parser.add_argument('--num_objects', type=int, default=1)
    parser.add_argument('--num_rot_anchors', type=int, default=60)
    parser.add_argument('--num_depth_pixels', type=int, default=500)
    # train
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--test_per_num_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--noise_trans', default=0.01, help='random noise added to translation')
    parser.add_argument('--num_mesh_points', type=int, default=500)
    opt = parser.parse_args()
    # result
    opt.result_dir = os.path.join(opt.output_dir, f'{time.strftime("%m%d_%H%M")}_{opt.part}_{opt.lr}')
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.result_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(opt.result_dir, 'models'), exist_ok=True)
    print("Options:")
    for k, v in opt.__dict__.items():
        print(f'{k}: {v}')
    print("-" * 20)
    return opt


def seed(seed):
    if not seed is None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Seed is set to be {seed}.")


def init(opt):
    run = SimpleNamespace()
    run.dataset = PoseDataset('train', opt)
    run.test_dataset = PoseDataset('test', opt)
    run.dataloader = torch.utils.data.DataLoader(run.dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers)
    run.test_dataloader = torch.utils.data.DataLoader(run.test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    run.diameter = run.dataset.get_diameter()
    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<')
    print('length of the training set: {0}'.format(len(run.dataset)))
    print('length of the testing set: {0}'.format(len(run.test_dataset)))
    print('diameter of the part: {0} m'.format(run.diameter))
    # network
    run.estimator = PoseNet(num_points=opt.num_depth_pixels, num_obj=opt.num_objects, num_rot=opt.num_rot_anchors)
    run.estimator.cuda()
    # loss
    run.sym_list = []
    run.criterion = Loss(run.sym_list, run.estimator.rot_anchors)
    # learning rate decay
    run.lr = opt.lr
    run.first_decay_thresh = 0.04
    run.first_decay_factor = 0.5
    run.first_decay_start = False
    run.second_decay_thresh = 0.02
    run.second_decay_factor = 0.5
    run.second_decay_start = False
    # optimizer
    run.optimizer = torch.optim.Adam(run.estimator.parameters(), lr=run.lr)
    # metric
    run.best_metric = np.Inf
    run.metric_history = {}
    # others
    run.index = torch.tensor([0], dtype=int).cuda()  # fill in the object index when passing to estimator and loss function
    run.st_time = time.time()
    return run 


def train_one_epoch(epoch, run):
    logger = setup_logger(f'epoch{epoch:02d}', os.path.join(opt.result_dir, 'logs', f'epoch_{epoch:02d}_train_log.txt'))
    logger.info(f'Train time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - run.st_time))}, Training started')
    run.estimator.train()
    train_count = 0
    avg_loss = 0.0
    avg_loss_r = 0.0
    avg_loss_t = 0.0
    avg_loss_reg = 0.0
    run.optimizer.zero_grad()
    for i, data in enumerate(run.dataloader):
        if i >= 100:
            break
        points, choose, img, target_t, target_r, model_points, gt_t = data
        points, choose, img, target_t, target_r, model_points, gt_t = \
            points.cuda(), choose.cuda(), img.cuda(), target_t.cuda(),target_r.cuda(), model_points.cuda(), gt_t.cuda()
        # estimate
        pred_r, pred_t, pred_c = run.estimator(img, points, choose, run.index)
        # loss
        loss, loss_r, loss_t, loss_reg = run.criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, \
            run.index, run.diameter)
        loss /= opt.batch_size
        loss.backward()
        avg_loss += loss.item()
        avg_loss_r += loss_r.item() / opt.batch_size
        avg_loss_t += loss_t.item() / opt.batch_size
        avg_loss_reg += loss_reg.item() / opt.batch_size
        train_count += 1
        if train_count % opt.batch_size == 0:  # batch learning
            run.optimizer.step()
            run.optimizer.zero_grad()
            logger.info(f'Train time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time()-run.st_time))} Epoch {epoch}' + \
                f'Batch {train_count // opt.batch_size} Frame {train_count} \tAvg_loss:{avg_loss}' +  \
                f'Avg_loss_r: {avg_loss_r} Avg_loss_t: {avg_loss_t} Avg_loss_reg: {avg_loss_reg} ')
            avg_loss = 0.0
            avg_loss_r = 0.0
            avg_loss_t = 0.0
            avg_loss_reg = 0.0

    print(f'>>>>>>>>----------epoch {epoch} train finish---------<<<<<<<<')


def transform(pred_r, pred_t, pred_c, points, target_r, model_points, gt_t):
    assert len(pred_c.shape) == 1, pred_c.shape
    assert len(pred_r.shape) == 2 and pred_r.shape[1] == 4, pred_r.shape
    assert len(points.shape) == 3 and points.shape[0] == 1 and points.shape[2] == 3, f'{points.shape}'
    assert len(pred_t.shape) == 3 and pred_t.shape[0] == 1 and pred_t.shape[2] == 3, f'{pred_t.shape}'
    print(len(model_points.shape), model_points.shape[1])
    assert len(model_points.shape) == 2 and model_points.shape[1] == 3, model_points.shape
    assert len(target_r.shape) == 2 and target_r.shape[1] == 3, f'{target_r.shape}'
    assert gt_t.shape == 3
    
    # transform
    print(pred_c)
    how_min, which_min = torch.min(pred_c, dim=0)
    print(how_min, which_min)
    pred_r = pred_r[which_min].cpu().data.numpy()
    pred_r = quaternion_matrix(pred_r)[:3, :3]
    pred_r = torch.tensor(pred_r).cuda()
    try:
        pred_t, _ = ransac_voting_layer(points, pred_t)
    except:
        print("RANSAC failed. Skipped.")
        return None
    pred_ps = np.matmul(model_points, pred_r.T) + pred_t
    target_ps = target_r + gt_t
    return pred_ps, target_ps, how_min


def compute_metric(run, pred_ps, target_ps, pred_t, target_t, pred_r, target_r):
    assert len(pred_ps.shape) == 2
    assert pred_ps.shape[1] == 3
    assert len(target_ps.shape) == 2
    assert target_ps.shape[1] == 3
    assert pred_t.shape == (3, 1)
    assert target_t.shape == (3, 1)
    assert pred_r.shape == (3, 3)
    assert target_r.shape == (3, 3)

    metric = {}
    metric["dist_error"] = torch.mean(torch.linalg.norm(pred_ps - target_ps, dim=1)).item()
    metric["adj_dist_error"] = metric["dist_error"] / run.diameter

    metric["t_error"] = torch.linalg.norm(pred_t - target_t)
    metric["adj_t_error"] = metric["t_error"] / run.diameter

    diff_r = torch.matmul(target_r, pred_r.T)  # use target_r = diff_r * pred_r
    diff_angle = np.rad2deg(np.arccos((np.trace(diff_r) - 1) / 2))  #
    metric["angle_error"] = diff_angle

    return metric


def record_metric(single_metric, average_metric):
    success_thresholds = {"adj_dist_error": [0.05, 0.1, 0.2, 0.5], "adj_t_error": [0.05, 0.1, 0.2, 0.5], \
        "angle_error": [5, 10, 20, 30]}
    # value
    for k, v in single_metric.items():
        average_metric[k] = average_metric.get(k, 0) + v
    # count
    for m, thresh in success_thresholds.items():
        for t in thresh:
            k = str(t) + '_' + m
            if k not in average_metric.keys():  # make sure every key exists when printing
                average_metric[k] = 0
            if single_metric[m] < t:
                average_metric[k] += 1

    return average_metric


def test_one_epoch(epoch, run):
    logger = setup_logger(f'epoch{epoch:02d}_test', os.path.join(opt.result_dir, 'logs', f'epoch_{epoch:02d}_test_log.txt'))
    logger.info(f'Test time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - run.st_time))}, Testing started')
    run.estimator.eval()
    # total metric
    test_count = 0
    average_metric = {}
    for i, data in enumerate(run.test_dataloader):
        if i >= 100:
            break
        points, choose, img, target_t, target_r, model_points, gt_t = data
        points, choose, img, target_t, target_r, model_points, gt_t = \
            points.cuda(), choose.cuda(), img.cuda(), target_t.cuda(), target_r.cuda(), model_points.cuda(), gt_t.cuda()
        with torch.no_grad():
            pred_r, pred_t, pred_c = run.estimator(img, points, choose, run.index)
        pred_r, pred_t, pred_c = pred_r.detach(), pred_t.detach(), pred_c.detach()
        loss, _, _, _ = run.criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, \
            run.index, run.diameter)
        # evalaution
        ps = transform(pred_r[0], pred_t, pred_c[0], points.detach(), target_r[0], model_points[0].detach(), gt_t[0])
        if ps is None:
            continue
        pred_ps, target_ps, how_min = ps
        # metric
        single_metric = compute_metric(run, pred_ps, target_ps, pred_t, target_t, pred_r, target_r)
        average_metric = record_metric(single_metric, average_metric)
        logger.info(f"Test time {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - run.st_time))} Epoch {epoch} \
                    Test Frame {test_count} Avg_loss:{loss:f} confidence:{how_min:f} distance:{single_metric['dist_error']:f}")
        test_count += 1
        
    # log results for this test epoch
    logger.info(f'Test time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - run.st_time))} \
        Epoch {epoch} {opt.part} TEST FINISH ')
    # compute average   
    for k, v in average_metric.items():
        average_metric[k] = v / test_count 
        logger.info(f'{k}: {average_metric[k]}')
    run.metric_history[f'epoch_{epoch:02d}'] = average_metric
    model_save_path = os.path.join(opt.result_dir, 'models', f'pose_model_{epoch:02d}_{average_metric["dist_error"]:06f}.pth')
    torch.save(run.estimator.state_dict(), model_save_path)

    # judge best metric
    if average_metric['dist_error'] < run.best_metric:
        run.best_metric = average_metric['dist_error']
        with open(os.path.join(opt.result_dir, 'best_models.txt'), 'w') as f:
            f.write(model_save_path)
    # adjust learning rate if necessary
    if run.best_metric < run.first_decay_thresh and not run.first_decay_start:
        run.first_decay_start = True
        run.lr *= run.first_decay_factor
        run.optimizer = torch.optim.Adam(run.estimator.parameters(), lr=run.lr)
    if run.best_metric < run.second_decay_thresh and not run.second_decay_start:
        run.second_decay_start = True
        run.lr *= run.second_decay_factor
        run.optimizer = torch.optim.Adam(run.estimator.parameters(), lr=run.lr)

    print(f'>>>>>>>>----------epoch {epoch} test finish---------<<<<<<<<')

def init_eval(opt):
    pass

def project_image():
    pass

def eval(run):
    pass


if __name__ == '__main__':
    opt = options()
    seed(opt.seed)

    # randomly_split(opt)

    run = init(opt)
    for epoch in range(opt.num_epoch):
        train_one_epoch(epoch, run)
        if epoch % opt.test_per_num_epoch == 0:
            test_one_epoch(epoch, run)
    del run

    # run = init_eval(opt)
    # eval(run)
