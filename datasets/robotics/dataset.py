import random
import numpy as np
import numpy.ma as ma
import yaml
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import json
import open3d as o3d
import h5py

image_height = 1200
image_width = 1920

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans):
        self.objlist = [
            1, # SF-CJd60-097-016-016
            2, # SF-CJd60-097-026
            3, # SongFeng_0005
            4, # SongFeng_306
            5, # SongFeng_311
            6, # SongFeng_318
            7, # SongFeng_332
            8, # 21092302
            9, # 6010018CSV
            10 # 6010022CSV
        ]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []  # mask
        self.list_obj = []
        self.list_rank = []
        self.list_meta = []
        self.pt = {}
        self.root = root
        self.diameters = {}
        self.mask = {}

        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                # if self.mode == 'test' and item_count % 10 != 0:
                #     continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                image = input_line.split('_')[0]
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, image))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, image))

                self.list_obj.append(item)
                self.list_rank.append(input_line)
                self.list_meta.append('{0}/data/{1}/label/{2}.json'.format(self.root, '%02d' % item, image))

            self.pt[item] = extract_model_pcd('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            self.mask[item] = '{0}/data/{1}/mask.hdf5'.format(self.root, '%02d' % item)
            with open('{0}/models/models_info.yml'.format(self.root), 'r') as meta_file:
                model_info = yaml.safe_load(meta_file)
                self.diameters[item] = model_info[item]['diameter'] / 1000.0

            print('Object {0} buffer loaded'.format(item))

        self.length = len(self.list_rgb)

        self.xmap = np.array([[i for i in range(image_width)] for j in range(image_height)])
        self.ymap = np.array([[j for i in range(image_width)] for j in range(image_height)])
        
        self.num = num
        self.symmetry_obj_idx = []
        self.num_pt_mesh = 500

        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.noise_trans = noise_trans
        

    def __getitem__(self, index):
        obj = self.list_obj[index]
        rank = self.list_rank[index]
        idx_image, idx_instance = rank.split('_')[0], rank.split('_')[1]

        img = Image.open(self.list_rgb[index])  # color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth = np.array(Image.open(self.list_depth[index]))  # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        with h5py.File(self.mask[obj], 'r') as h5_file:
            # label = np.array(h5_file[f'{idx_image}/{idx_instance}'][:])
            temp_idx_instance = idx_instance.lstrip('0') if idx_instance.lstrip('0') else '0'
            label = np.array(h5_file[f'{idx_image}/{temp_idx_instance}'][:])
        
        
        with open(self.list_meta[index]) as f:
            meta = json.load(f)
        cam_fx = meta["K"][0][0]
        cam_fy = meta["K"][1][1]
        cam_cx = meta["K"][0][2]
        cam_cy = meta["K"][1][2]

        # mask for both valid depth and foreground
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(True)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, True))
        mask = mask_label * mask_depth

        if self.add_noise:
            img = self.trancolor(img)
        img_masked = np.array(img)

        rmin, rmax, cmin, cmax = get_bbox(meta, idx_instance)  # choose predifined box size 
        img_masked = img_masked[rmin:rmax, cmin:cmax, :3]

        cam_scale = 1000.0
        depth_scale = 22.0  # temp
        target_r = np.resize(np.array(meta['instances'][idx_instance]['R']), (3, 3))
        target_t = np.array(meta['instances'][idx_instance]['t']) / cam_scale

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]  # index in the box
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

        # point cloud
        pt2 = depth_masked / cam_scale / depth_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise:
            # shift
            add_t = np.random.uniform(-self.noise_trans, self.noise_trans, (1, 3))
            target_t = target_t + add_t
            # jittering
            add_t = add_t + np.clip(0.001*np.random.randn(cloud.shape[0], 3), -0.005, 0.005)
            cloud = np.add(cloud, add_t)
        # position target
        gt_t = target_t
        target_t = target_t - cloud
        target_t = target_t / np.linalg.norm(target_t, axis=1)[:, None]

        # rotation target
        model_points = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh)
        model_points = np.delete(model_points, dellist, axis=0)
        target_r = np.dot(model_points, target_r.T)

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.transform(img_masked), \
               torch.from_numpy(target_t.astype(np.float32)), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)]), \
               torch.from_numpy(gt_t.astype(np.float32))


    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh

    def get_diameter(self):
        return self.diameters


def get_bbox(meta, idx_instance):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    instance_info = meta['instances'][idx_instance]
    bbx = [instance_info['y_min'], instance_info['y_max'], instance_info['x_min'], instance_info['x_max']]
    image_h, image_w = int(meta['image_height']), int(meta['image_width'])
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


def extract_model_pcd(model_path):
    print('extract model point cloud from:', model_path)
    pcd = o3d.io.read_point_cloud(model_path)
    pcd = np.array(pcd.points)
    print('point cloud shape:', pcd.shape)
    # ones = np.ones((pcd.shape[0], 1))
    # pcd_one = np.concatenate([pcd, ones], axis=1)
    return pcd
