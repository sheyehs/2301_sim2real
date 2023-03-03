import os
import argparse
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from dataset import PoseDataset 
from lib.network import PoseNet
from lib.loss import Loss
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.transformations import quaternion_matrix
from lib.utils import setup_logger
import debugging

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', type=str, default='./split')
    parser.add_argument('--part', type=str, help='part name')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--noise_trans', default=0.01, help='random noise added to translation')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--nepoch', type=int, default=50, help='max number of epochs to train')
    return parser.parse_args()

def main(opt):
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    opt.dataset_path = './data.hdf5'
    opt.num_objects = 1
    opt.result_dir = f'results/{time.strftime("%m%d_%H%M")}_{opt.part}_lr_{opt.lr}'
    opt.repeat_epoch = 1  # 10
    opt.num_rot = 60
    opt.num_depth_pixels = 500
    opt.num_mesh_points = 500
    opt.split_train_file = 'train.txt'
    opt.split_test_file = 'test.txt'

    dataset = PoseDataset('train', opt)
    test_dataset = PoseDataset('test', opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    sym_list = dataset.get_sym_list()
    diameter = dataset.get_diameter()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<')
    print('length of the training set: {0}'.format(len(dataset)))
    print('length of the testing set: {0}'.format(len(test_dataset)))
    print('number of sample points on mesh: {0}'.format(opt.num_mesh_points))
    print('symmetrical object list: {0}'.format(sym_list))

    os.makedirs(opt.result_dir, exist_ok=True)
    writer = SummaryWriter(opt.result_dir) 
    
    # network
    estimator = PoseNet(num_points=opt.num_depth_pixels, num_obj=opt.num_objects, num_rot=opt.num_rot)
    estimator.cuda()
    # loss
    criterion = Loss(sym_list, estimator.rot_anchors)
    # learning rate decay
    best_test = np.Inf
    opt.first_decay_start = False
    opt.second_decay_start = False
    # optimizer
    optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.lr)
    global_step = 0
    idx = torch.tensor([0], dtype=int).cuda()
    # train
    st_time = time.time()
    for epoch in range(opt.nepoch):
        logger = setup_logger('epoch%02d' % epoch, os.path.join(opt.result_dir, 'epoch_%02d_train_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        estimator.train()
        train_count = 0
        train_loss_avg = 0.0
        train_loss_r_avg = 0.0
        train_loss_t_avg = 0.0
        train_loss_reg_avg = 0.0
        optimizer.zero_grad()
        for rep in range(opt.repeat_epoch):
            lp = debugging.line_printer()
            delta_times = [0.0] * 9
            for i, data in enumerate(dataloader):
                if i >= 10000:
                    for j, t in enumerate(delta_times):
                        print(j, '\t', t)
                        exit()
                delta_times[0] += lp.print_line()  ###
                points, choose, img, target_t, target_r, model_points, gt_t = data
                points, choose, img, target_t, target_r, model_points = Variable(points).cuda(), \
                                                                             Variable(choose).cuda(), \
                                                                             Variable(img).cuda(), \
                                                                             Variable(target_t).cuda(), \
                                                                             Variable(target_r).cuda(), \
                                                                             Variable(model_points).cuda()
                delta_times[1] += lp.print_line()  ###
                pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
                delta_times[2] += lp.print_line()  ###
                loss, loss_r, loss_t, loss_reg = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, diameter)
                delta_times[3] += lp.print_line()  ###
                # torch.cuda.empty_cache()
                loss.backward()
                delta_times[4] += lp.print_line()  ###
                train_loss_avg += loss.item()
                train_loss_r_avg += loss_r.item()
                train_loss_t_avg += loss_t.item()
                train_loss_reg_avg += loss_reg.item()
                train_count += 1
                delta_times[5] += lp.print_line()  ###
                if train_count % opt.batch_size == 0:
                    delta_times[6] += lp.print_line()  ###
                    global_step += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    delta_times[7] += lp.print_line()  ###
                    # write results to tensorboard
                    writer.add_scalars(
                        "log", {
                            'learning_rate': opt.lr,
                            'loss': train_loss_avg / opt.batch_size,
                            'loss_r': train_loss_r_avg / opt.batch_size,
                            'loss_t': train_loss_t_avg / opt.batch_size,
                            'loss_reg': train_loss_reg_avg / opt.batch_size
                        },
                        global_step
                    )
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_loss:{4:f}'.format(time.strftime("%Hh %Mm %Ss", 
                        time.gmtime(time.time()-st_time)), epoch, int(train_count/opt.batch_size), train_count, train_loss_avg/opt.batch_size))
                    train_loss_avg = 0.0
                    train_loss_r_avg = 0.0
                    train_loss_t_avg = 0.0
                    train_loss_reg_avg = 0.0
                    delta_times[8] += lp.print_line()  ###

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%02d_test' % epoch, os.path.join(opt.result_dir, 'epoch_%02d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        estimator.eval()
        test_dis = 0.0
        test_count = 0
        success_count = np.zeros(4, dtype=int)

        for j, data in enumerate(testdataloader):
            points, choose, img, target_t, target_r, model_points, gt_t = data
            points, choose, img, target_t, target_r, model_points = Variable(points).cuda(), \
                                                                         Variable(choose).cuda(), \
                                                                         Variable(img).cuda(), \
                                                                         Variable(target_t).cuda(), \
                                                                         Variable(target_r).cuda(), \
                                                                         Variable(model_points).cuda(), \

            with torch.no_grad():
                pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
            loss, _, _, _ = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, diameter)
            # evalaution
            how_min, which_min = torch.min(pred_c, 1)
            pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
            pred_r = quaternion_matrix(pred_r)[:3, :3]
            try:
                pred_t, pred_mask = ransac_voting_layer(points, pred_t)
            except:
                print("RANSAC failed. Skipped.")
                continue
            pred_t = pred_t.cpu().data.numpy()
            model_points = model_points[0].cpu().detach().numpy()
            pred = np.dot(model_points, pred_r.T) + pred_t
            target = target_r[0].cpu().detach().numpy() + gt_t[0].cpu().data.numpy()
            if idx[0].item() in sym_list:
                pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
                target = torch.index_select(target, 1, inds.view(-1) - 1)
                dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
            else:
                dis = np.mean(np.linalg.norm(pred - target, axis=1))
            logger.info('Test time {0} Test Frame No.{1} loss:{2:f} confidence:{3:f} distance:{4:f}'.format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss, how_min[0].item(), dis))
            error_ratio = dis / diameter
            cond = [error_ratio < 0.05, error_ratio < 0.1, error_ratio < 0.2, error_ratio < 0.5] 
            success_count[cond] += 1
            test_count += 1
            test_dis += dis
        # compute accuracy
        accuracy = success_count / test_count
        test_dis = test_dis / test_count        
        # log results
        logger.info('')
        logger.info('Test time {0} Epoch {1} {2} TEST FINISH '.format(time.strftime("%Hh %Mm %Ss",
            time.gmtime(time.time() - st_time)), epoch, opt.part))
        logger.info(f'accuracy of 0.05 diameter: {accuracy[0]}')
        logger.info(f'accuracy of 0.1 diameter: {accuracy[1]}')
        logger.info(f'accuracy of 0.2 diameter: {accuracy[2]}')
        logger.info(f'accuracy of 0.5 diameter: {accuracy[3]}')

        logger.info(f'average distance error: {test_dis}')
        logger.info(f'average t error:')  # todo
        logger.info(f'average R error:')  # todo
        # tensorboard
        # writer.add_scalars(
        #                 "log",{
        #                     'accuracy': accuracy,
        #                     'test_dis': test_dis
        #                 },
        #                 global_step
        #             )
        # save model
        best_test = min(test_dis, best_test)
        torch.save(estimator.state_dict(), '{0}/pose_model_{1:02d}_{2:06f}.pth'.format(opt.result_dir, epoch, best_test))
        # adjust learning rate if necessary
        if best_test < 0.04 and not opt.first_decay_start:  # 0.016
            opt.first_decay_start = True
            opt.lr *= 0.5
            optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.lr)
        if best_test < 0.02 and not opt.second_decay_start:  # 0.013
            opt.second_decay_start = True
            opt.lr *= 0.5
            optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.lr)

        print('>>>>>>>>----------epoch {0} test finish---------<<<<<<<<'.format(epoch))


if __name__ == '__main__':
    opt = options()
    main(opt)
