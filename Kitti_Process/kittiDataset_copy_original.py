import sys
import os
import math
from builtins import int

import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import torchvision
from Kitti_Process.kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar
from Kitti_Process.kitti_bev_utils import makeBEVMap, drawRotatedBox, get_corners,lidar_to_top
import Kitti_Process.transformation as transformation
import Kitti_Process.kitti_fov_utils as fov_utils 
from config import kitti_config as cnf
#from Kitti_Process.kittiUtils import compute_box_3d, project_to_image
from utils.visualization_utils import compute_box_3d, project_to_image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

BOX_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3], [1, 6], [2, 5]]
IMG_WIDTH, IMG_HEIGHT = 1242, 375



class KittiDataset(Dataset):
    def __init__(self, configs, mode='train', lidar_aug=None, hflip_prob=None):
        self.dataset_dir = configs.dataset_dir
        self.hm_size = configs.hm_size
        self.imageSize = configs.imageSize
        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects
        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        if self.mode == 'test':
            sub_folder = 'testing'
        sub_folder = 'training'
        self.lidar_aug = lidar_aug
        self.hflip_prob = hflip_prob
        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")
        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne")
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()]
        self.num_samples = len(self.sample_id_list)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        if self.mode == 'test':
            return self.load_img_only(index)
        else:        
            return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""
        bev_map = self.get_BEV(index)
        fov_maps = self.get_FOV(index)
        img_rgb = cv2.resize(img_rgb, dsize=(self.imageSize[1],self.imageSize[0]), interpolation=cv2.INTER_LINEAR)
        bev_map = torch.from_numpy(bev_map[0])
        fov_maps = torch.from_numpy(fov_maps[0])
        img_rgb = torch.from_numpy(img_rgb)
        return bev_map, img_rgb, fov_maps

    def load_img_with_targets(self, index):
        sample_id = int(self.sample_id_list[index])
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
        lidarData = self.get_lidar(sample_id)
        img_path, img_rgb = self.get_image(sample_id)
        #print("image path is: ",img_path)
        calib = self.get_calib(sample_id)
        labelsM, has_labels, orig_label = self.get_label(sample_id)
        bev_map = self.get_BEV(index)
        fov_maps = self.get_FOV(index)
        img_rgb = cv2.resize(img_rgb, dsize=(self.imageSize[1],self.imageSize[0]), interpolation=cv2.INTER_LINEAR)
        bev_map = torch.from_numpy(bev_map[0]) #torch.Tensor([torch.from_numpy(bevs) for bevs in bev_map])
        fov_maps = torch.from_numpy(fov_maps[0])
        img_rgb = torch.from_numpy(img_rgb)
        gt_boxes_all = []
        gt_idxs_all = []
        gt_idxs = []
        for obj in orig_label:
            if obj[0] not in cnf.CLASS_NAME_TO_ID.keys():
                continue
            res= compute_box_3d(obj)
            pts = project_to_image(res, calib.P2)
            for i in range(pts.shape[0]):
                for j in range(pts.shape[1]):
                    if pts[i][j] < 0 :
                        pts[i][j] = 0
            pts[:,0] = pts[:,0] * (img_rgb.shape[1]/ 1242)
            pts[:,1] = pts[:,1] * (img_rgb.shape[0]/ 375)  
            obj[4] = int(float(obj[4])) * (img_rgb.shape[1]/ 1242)
            obj[6] = int(float(obj[6])) * (img_rgb.shape[1]/ 1242)
            obj[5] = int(float(obj[5])) * (img_rgb.shape[0]/ 375)
            obj[7] = int(float(obj[7])) * (img_rgb.shape[0]/ 375)
            pts2d = [obj[4],obj[5],obj[6],obj[7]]
            gt_boxes_all.append(torch.from_numpy(pts))
            gt_idxs.append(int(cnf.CLASS_NAME_TO_ID[obj[0]]))
        gt_boxes_all = torch.stack(gt_boxes_all)    
        gt_idxs = torch.Tensor(gt_idxs)
        return img_rgb, bev_map, fov_maps, gt_boxes_all,gt_idxs 

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        bevs = list()
        fovs = list()
        for b in batch:
            images.append(b[0])
            bevs.append(b[1])
            fovs.append(b[2])
            if self.mode != 'test':
                boxes.append(b[3])
                labels.append(torch.Tensor(b[4]))
        images = torch.stack(images, dim=0)
        bevs = torch.stack(bevs, dim=0)
        fovs = torch.stack(fovs, dim=0)
        if self.mode != 'test':
            boxes = pad_sequence(boxes, batch_first=True, padding_value=-1)
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        return images,bevs, fovs, boxes, labels

    def get_image(self, idx):
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return img_path, img

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return Calibration(calib_file)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_BEV(self, idx):
        dtype = np.float32
        n_vec = 4 
        pc_vel = self.get_lidar(idx)
        top_view = lidar_to_top(pc_vel)
        reshaped_topview=top_view.reshape(top_view.shape[0],-1)
        #np.savetxt(output_dir+str(data_idx)+'.txt',reshaped_topview)
        def draw_top_image(lidar_top):
            top_image = np.sum(lidar_top, axis=2)
            top_image = top_image - np.min(top_image)
            divisor = np.max(top_image) - np.min(top_image)
            top_image = top_image / divisor * 255
            top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
            return top_image
        top_img = draw_top_image(top_view)
        return top_view,top_img
    
    def get_FOV(self, idx, gt_boxes=None, pred_boxes=None):
        sample_id = int(self.sample_id_list[idx])
        img_path, img = self.get_image(sample_id) 
        calib_path = os.path.join(self.calib_dir, '{:06d}.txt'.format(sample_id))       
        V2C, R0, P2 = fov_utils.get_calib(calib_path)
        calibObj = self.get_calib(idx)
        V2C_t, R0_t, P2_t = calibObj.V2C, calibObj.R0, calibObj.P2
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(sample_id))
        pts, ref = fov_utils.get_velo(lidar_file,calib_path)
        img = None
        if img is not None:
            img = np.copy(img)  # Clone
        else:
            img = np.zeros((375, 1242, 1))
        if pts.shape[0] != 3:
            pts = pts.T
        if ref.shape[0] != 1:
            ref = ref.T

        def draw_boxes_2D(boxes, color):
            for box in boxes:
                cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 1)

        def draw_boxes_3D(boxes, color):
            for box in boxes:
                corners = fov_utils.project(P2, box.get_corners()).astype(np.int32)
                for start, end in BOX_CONNECTIONS:
                    x1, y1 = corners[:, start]
                    x2, y2 = corners[:, end]
                    cv2.line(img, (x1, y1), (x2, y2), color, 1)

        out_type=['height','depth', 'intensity']
        if pts is not None:
            pts_projected = fov_utils.project(P2, pts).astype(np.int32).T
            image_depth = np.copy(img)
            image_tensity = np.copy(img)
            image_height = np.copy(img)
            for i in range(pts_projected.shape[0]):
                if 'depth' in out_type:
                    clr = pts.T[i][2] / 70.
                    cv2.circle(image_depth, (pts_projected[i][0], pts_projected[i][1]), 4, float(clr), -1)
                if 'intensity' in out_type:                    
                    clr = ref.T[i][0]
                    cv2.circle(image_tensity, (pts_projected[i][0], pts_projected[i][1]), 4, float(clr), -1)
                if 'height' in out_type:                    
                    clr = 1. - (pts.T[i][1] + 1.) / 4.
                    cv2.circle(image_height, (pts_projected[i][0], pts_projected[i][1]), 4, float(clr), -1)
        return [image_depth, image_tensity, image_height]

    def get_label(self, idx):
        labels = []
        label_path = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        orig_labels = []
        for line in open(label_path, 'r'):
            line = line.rstrip()
            line_parts = line.split(' ')
            obj_name = line_parts[0]
            if obj_name not in cnf.CLASS_NAME_TO_ID.keys():
                continue
            cat_id = int(cnf.CLASS_NAME_TO_ID[obj_name])
            if cat_id <= -99: 
                continue
            truncated = int(float(line_parts[1]))
            occluded = int(line_parts[2])
            alpha = float(line_parts[3]) 
            bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])])
            h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10])
            x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13])
            ry = float(line_parts[14])  
            object_label = [cat_id, x, y, z, h, w, l, ry]
            labels.append(object_label)
            orig_labels.append(line_parts)
        if len(labels) == 0:
            labels = np.zeros((1, 8), dtype=np.float32)
            has_labels = False
        else:
            labels = np.array(labels, dtype=np.float32)
            has_labels = True
        return labels, has_labels, orig_labels

    def draw_img_with_label(self, index):
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(sample_id)
        if has_labels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)
        if self.lidar_aug:
            lidarData, labels[:, 1:] = self.lidar_aug(lidarData, labels[:, 1:])
        lidarData, labels = get_filtered_lidar(lidarData, cnf.boundary, labels)
        bev_map = makeBEVMap(lidarData, cnf.boundary)
        return bev_map, labels, img_rgb, img_path

