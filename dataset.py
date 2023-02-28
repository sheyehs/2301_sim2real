import random
import numpy as np
import numpy.ma as ma
import yaml
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import open3d as o3d
import h5py
import os

image_height = 400
image_width = 640

class PoseDataset(data.Dataset):
    def __init__(self, mode, opt, split_file, add_noise: bool, noise_trans):
        """
        mode: train, test, eval, teacher, student
        """
        self.mode = mode
        self.dataset_path = opt.dataset_path
        self.split_dir = opt.split_dir
        self.split_file = split_file
        self.model_root = './models'

        self.part = opt.part

        
        print("Dataset mode is:", self.mode)
        if self.mode == 'student':
            self.label_dir = label_dir

        self.cam_scale = 1000.0
        self.depth_scale = 22.0  # to do: change

        self.list_image = []
        self.list_path = []  # instance number
        self.pt = None
        self.diameters = None

        item_count = 0
        # collect index
        file_name =  os.path.join(self.split_dir, self.part, split_file)
        if not os.path.isfile(file_name):
            print(f'The split list of part {self.part} does not exist! Skipped.')
            return
        input_file = open(file_name, 'r')
        while 1:
            item_count += 1
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]

            self.list_path.append(input_line)
            self.list_image.append(input_line[:input_line.rfind('/')])
        input_file.close()

        # read pcd
        model_path = f'{self.model_root}/{self.part}.master.ply'
        print('extract model point cloud from:', model_path)
        pcd = o3d.io.read_point_cloud(model_path)
        pcd = np.array(pcd.points)
        print('point cloud shape:', pcd.shape)
        self.pt = pcd

        # read diameter
        with open(f'{self.model_root}/diameters.yml', 'r') as meta_file:
            model_info = yaml.safe_load(meta_file)
            diameter = float(model_info[self.part]) / self.cam_scale
            print(f'read {self.part} diameter {diameter} m.')
            self.diameter = diameter

        print('Object {0} buffer loaded'.format(self.part))

        self.length = len(self.list_path)

        self.xmap = np.array([[i for i in range(image_width)] for j in range(image_height)])
        self.ymap = np.array([[j for i in range(image_width)] for j in range(image_height)])
        
        self.num = opt.num_depth_pixels  # number of points for predicting t
        self.num_pt_mesh = opt.num_mesh_points  # number of points for evaluating R
        self.symmetry_obj_idx = []

        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.noise_trans = noise_trans
        

    def __getitem__(self, index):
        instance_path = self.list_path[index]  # image_instance

        h = h5py.File(self.dataset_path, 'r')

        img = Image.fromarray(np.array(h[f'{self.list_image[index]}/color']))  
        depth = np.array(h[f'{self.list_image[index]}/depth'])
        label = np.array(h[f'{self.list_path[index]}/mask'])
          
        K = np.array(h[f'{self.list_image[index]}/K'])
        cam_fx = K[0][0]
        cam_fy = K[1][1]
        cam_cx = K[0][2]
        cam_cy = K[1][2]

        # mask for both valid depth and foreground
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, True))
        mask = mask_label * mask_depth

        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img)

        meta = h[self.list_path[index]]
        rmin, rmax, cmin, cmax = get_bbox(meta)  # choose predifined box size 
        img_masked = img[rmin:rmax, cmin:cmax, :3]

        if self.mode == 'student':
            h_label = h5py.File(self.label_dir, 'r')
            R_t = h_label[self.list_path[index]]
            target_r = np.array(R_t['R'])
            target_t = np.array(R_t['t']) / self.cam_scale
            target_t = target_t.transpose()
            h_label.close()
        elif self.mode =='teacher':
            pass
        else:
            target_r = np.array(meta['R'])
            target_t = np.array(meta['t']) / self.cam_scale

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]  # valid index in the box
        # return all zero vector if there is no valid point
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)
        # downsample points if there are too many
        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        # repeat points if not enough
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        # point cloud from depth 
        pt2 = depth_masked / self.cam_scale / self.depth_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)  # (n_points, 3)
        if self.add_noise:
            # shift
            add_t = np.random.uniform(-self.noise_trans, self.noise_trans, (1, 3))
            if self.mode == 'teacher':  # no target_t
                pass
            else:
                target_t = target_t + add_t
            # jittering
            add_t = add_t + np.clip(0.001*np.random.randn(cloud.shape[0], 3), -0.005, 0.005)
            cloud = np.add(cloud, add_t)
        # position target
        if self.mode == 'teacher':
            pass
        else:
            gt_t = target_t
            target_t = target_t - cloud  # relative vectors from depth point to target center 
            target_t = target_t / np.linalg.norm(target_t, axis=1)[:, None]  # normalize

        # rotation target
        model_points = self.pt / self.cam_scale
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh)
        model_points = np.delete(model_points, dellist, axis=0)

        if self.mode == 'teacher':
            pass
        else:
            target_r = np.dot(model_points, target_r.T)

        h.close()

        if self.mode == 'teacher':
            # no target_R, target_t and gt_t
            ret = [
                torch.from_numpy(cloud.astype(np.float32)),
                torch.LongTensor(choose.astype(np.int32)),
                self.transform(img_masked),
                torch.from_numpy(model_points.astype(np.float32)),
                torch.LongTensor([obj]),
                instance_path, 
                K, 
                img.shape[:2], 
                torch.tensor([rmin, rmax, cmin, cmax], dtype=int), 
                mask_label[rmin:rmax, cmin:cmax]
            ]
            return ret

        elif self.mode == 'train' or self.mode == 'test' or self.mode == 'student':
            ret = [
                torch.from_numpy(cloud.astype(np.float32)),
                torch.LongTensor(choose.astype(np.int32)),
                self.transform(img_masked),
                torch.from_numpy(target_t.astype(np.float32)),
                torch.from_numpy(target_r.astype(np.float32)),
                torch.from_numpy(model_points.astype(np.float32)),
                torch.from_numpy(gt_t.astype(np.float32))
            ]
            return ret

        elif self.mode == 'eval':  # need pass K and image to prject pred points onto
            ret = [
                torch.from_numpy(cloud.astype(np.float32)),
                torch.LongTensor(choose.astype(np.int32)),
                self.transform(img_masked),
                torch.from_numpy(target_r.astype(np.float32)),
                torch.from_numpy(model_points.astype(np.float32)),
                torch.from_numpy(gt_t.astype(np.float32)),
                instance_path,
                K,
                np.array(img)
            ]
            return ret

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh

    def get_diameter(self):
        return self.diameter
        
    def get_pcd_list(self):
        return self.pt


def get_bbox(meta):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    bbx = [np.array(meta['y_min']), np.array(meta['y_max']), np.array(meta['x_min']), np.array(meta['x_max'])]
    image_h, image_w = image_height, image_width
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= image_h:
        bbx[1] = image_h-1
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= image_w:
        bbx[3] = image_w-1
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    # choose appropriate box size
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b <= border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b <= border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    # new boundary [a, b)
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > image_h:
        delt = rmax - image_h
        rmax = image_h
        rmin -= delt
    if cmax > image_w:
        delt = cmax - image_w
        cmax = image_w
        cmin -= delt
    return rmin, rmax, cmin, cmax



