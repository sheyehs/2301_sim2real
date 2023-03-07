from new_train import *

def new_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--seed', type=int)
    # data
    parser.add_argument('--part', type=str)
    parser.add_argument('--eval_dataset_path', type=str, default='./data_real.hdf5') 
    parser.add_argument('--split_dir', type=str, default='./split')
    parser.add_argument('--split_eval_file', type=str, default='eval_on_real.txt')
    parser.add_argument('--output_root', type=str, default='./results_eval')
    # model
    parser.add_argument('--num_objects', type=int, default=1)
    parser.add_argument('--num_rot_anchors', type=int, default=60)
    parser.add_argument('--num_depth_pixels', type=int, default=500)

    parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--noise_trans', default=0.01, help='random noise added to translation')
    parser.add_argument('--num_mesh_points', type=int, default=500)
    opt = parser.parse_args()
    assert opt.model_path
    # result
    opt.outpur_dir = os.path.join(opt.output_root, f'{time.strftime("%m%d_%H%M")}_{opt.part}')
    os.makedirs(opt.outpur_dir, exist_ok=True)
    print("=" * 20)
    print("Options:")
    print("-" * 20)
    for k, v in opt.__dict__.items():
        print(f'{k}: {v}')
    print("-" * 20)
    return opt


def new_init_eval(opt):
    run = SimpleNamespace()
    run.part = opt.part
    run.logger = setup_logger(f'eval', os.path.join(opt.outpur_dir, f'eval_log.txt'))
    run.eval_dataset = PoseDataset('eval', opt)
    run.eval_dataloader = torch.utils.data.DataLoader(run.eval_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    run.diameter = run.eval_dataset.get_diameter()
    run.logger.info('>>>>>>>>----------Dataset loaded!---------<<<<<<<<')
    run.logger.info('length of the evaluation set: {0}'.format(len(run.eval_dataloader)))
    run.logger.info('diameter of the part: {0} m'.format(run.diameter))
    # network
    run.estimator = PoseNet(num_points=opt.num_depth_pixels, num_obj=opt.num_objects, num_rot=opt.num_rot_anchors)
    run.estimator.load_state_dict(torch.load(opt.model_path))
    run.estimator.cuda()
    run.estimator.eval()
    # loss
    run.sym_list = []
    run.criterion = Loss(run.sym_list, run.estimator.rot_anchors)
    # others
    run.index = torch.tensor([0], dtype=int).cuda()  # fill in the object index when passing to estimator and loss function
    run.st_time = time.time()
    run.image_shape = (400, 640, 3)
    # output
    run.output_eval_images_dir = os.path.join(opt.outpur_dir, 'images')
    os.makedirs(run.output_eval_images_dir, exist_ok=True)
    run.logger.info(f'Eval time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - run.st_time))}, started')
    return run

if __name__ == '__main__':
    opt = new_options()
    seed(opt.seed)

    run = new_init_eval(opt)
    eval(run)
