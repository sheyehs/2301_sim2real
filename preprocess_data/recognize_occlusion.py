import numpy as np
import open3d as o3d
import os
import cv2
import matplotlib.pyplot as plt
import json
import h5py
from multiprocessing import pool, cpu_count
import multiprocessing
from functools import partial


def get_model(model_path):
    # read model points
    # Read the point cloud
    pcd = o3d.io.read_point_cloud(model_path)
    # Convert open3d format to numpy array
    pcd = np.array(pcd.points)
    ones = np.ones((pcd.shape[0],1))
    pcd_ = np.concatenate([pcd, ones], 1)
    return pcd_

# def get_info(file_path):
def get_info(depth_path, color_path, param_path, label_path):

    # read image
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # read json
    with open(param_path) as f:
        params = json.load(f)
    K = params['K']
    K = np.array(K)
    K = K.reshape((3,3))
    # read numpy
    labels = np.load(label_path)
    return color, K, labels, depth



def get_mask(labels, pcd):     # return a dictionary

    n_instances = len(labels)
    mask_all = {}
    for i in range(n_instances):
        pose_i = labels[i]
        points = np.matmul(pose_i, pcd.transpose())
        points = points.astype(int)
        # print(points.shape)       # [4,len]
        for j in range(len(points[0])):
            if points[0][j] in mask_all:
                if points[1][j] in mask_all[points[0][j]]:
                    if points[2][j] > mask_all[points[0][j]][points[1][j]][0]:
                        mask_all[points[0][j]][points[1][j]] = [points[2][j], i]
                    else:
                        points[2][j] = -1e5
                else:
                    mask_all[points[0][j]][points[1][j]] = [points[2][j], i]
            else:
                mask_all[points[0][j]] = {}
                mask_all[points[0][j]][points[1][j]] = [points[2][j], i]

    # pixels delete the overlapping ones
    cnt = 0
    for ele in list(mask_all.keys()):
        cnt+=len(list(mask_all[ele].keys()))

    return mask_all, cnt




def generate_mask_file(mask, color, K, cnt, n_instances):
    mask_with_label = np.zeros((cnt,4))
    now = 0
    for i in list(mask.keys()):
        for j in list(mask[i].keys()):
            mask_with_label[now] = [i,j,mask[i][j][0],mask[i][j][1]]
            now+=1
    color2 = color.copy()
    masking = mask_with_label.transpose()
    pixels_2d = np.matmul(K, masking[:3]/masking[2]).astype(int)[:2]       
    # (2, 116335) width range [352, 1246]; height range [33, 616]

    # generate mask for each model
    mask_final = np.zeros((n_instances, *color.shape[:2])) #(19, 1200, 1920)
    pixels_2d_mask = pixels_2d.transpose()
    for i, pixel in enumerate(pixels_2d_mask):
        if 0<=pixel[1]<mask_final.shape[1] and 0<=pixel[0]<mask_final.shape[2]:
            mask_final[int(mask_with_label[i][3])][pixel[1]][pixel[0]] = 1
    return mask_final



def generate_one_mask(depth_file, depth_path, images_path, labels_path, saving_path_prior, pcd):
    file_prior = depth_file.split(".")[0]       # "000000-003999"
    depth_file_path = os.path.join(depth_path, depth_file)
    images_file_path = os.path.join(images_path, file_prior+".jpg")
    param_file_path = os.path.join(images_path, file_prior+".json")
    labels_file_path = os.path.join(labels_path, file_prior+".npy")
    saving_file_path = os.path.join(saving_path_prior, file_prior+".h5")
    if os.path.exists(depth_file_path) and os.path.exists(images_file_path) and os.path.exists(param_file_path) and os.path.exists(labels_file_path):
        # get original info from docs
        color, K, labels, depth = get_info(depth_file_path, images_file_path, param_file_path, labels_file_path)
        n_instances = len(labels)

        # get all points that need mask
        mask, cnt = get_mask(labels, pcd)

        mask_to_file = generate_mask_file(mask, color, K, cnt, n_instances)     # [N, image size]
        # print(mask_to_file.shape, type(mask_to_file))
        # hf = h5py.File(saving_file_path, "w")
        # hf.create_dataset("mask", data=mask_to_file)
        # hf.create_dataset("label", data=labels)
        # hf.create_dataset("image", data=color)
        # hf.create_dataset("K", data=K)
        # hf.create_dataset("depth", data=depth)
        # hf.close()



if __name__ == '__main__':
    data_path = "WenyuHan/sim2real/2022-11-15"
    saving_path = "WenyuHan/sim2real/mask_gt_YFW"
    file_list = ["SongFeng", "Toyota","ZSRobot"]
    # mode = 0o666
    for path1 in file_list:                         # WenyuHan/sim2real/2022-11-15/SongFeng
        if not os.path.exists(os.path.join(saving_path, path1)):
            os.mkdir(os.path.join(saving_path, path1))    # WenyuHan/sim2real/mask_gt_YFW/SongFeng
        subpath = os.path.join(data_path, path1)
        subpath_list = os.listdir(subpath)
        for path2 in subpath_list:
            path = os.path.join(subpath, path2)     # WenyuHan/sim2real/2022-11-15/SongFeng/SF-CJd60-097-016-016
            if not os.path.exists(os.path.join(saving_path, path1, path2)):
                os.mkdir(os.path.join(saving_path, path1, path2))
            model_path = os.path.join(path, "part_info", path2+".master.ply")
            sub_file_list = os.listdir(path)
            for p in sub_file_list:                 
                if p[-9:]=="synthetic":
                    data_path = os.path.join(path, p)
            
                pcd = get_model(model_path)
            # filepath = data_path
            depth_path = os.path.join(data_path, "depth")
            images_path = os.path.join(data_path, "images")
            labels_path = os.path.join(data_path, "labels")
            depth_file_list = os.listdir(depth_path)
            saving_path_prior = os.path.join(saving_path, path1, path2)
            # for depth_file in depth_file_list:
                # generate_one_mask(depth_file, depth_path, images_path, labels_path, path1, path2)
            l = len(depth_file_list)
            p = pool.Pool(processes=cpu_count())
            f = partial(generate_one_mask, depth_path=depth_path, images_path=images_path, labels_path=labels_path, saving_path_prior=saving_path_prior, pcd=pcd)
            p.map(f, depth_file_list)
            p.close()