import math
import os
import sys

import cv2
import numpy as np
import config.kitti_config as cnf

IMG_WIDTH, IMG_HEIGHT = 1242, 375
GT_COLOR = (0.0, 1.0, 0.0)
PRED_COLOR = (1.0, 0.0, 0.0)
BOX_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3], [1, 6], [2, 5]]


def get_calib(path):
    # Read file 
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            # Skip if line is empty
            if len(line) == 0:
                continue
            # Load required matrices only
            matrix_name = line[0][:-1]
            if matrix_name == 'Tr_velo_to_cam':
                V2C = np.array([float(i) for i in line[1:]]).reshape(3, 4)  # Read from file
                V2C = np.insert(V2C, 3, values=[0, 0, 0, 1], axis=0)  # Add bottom row
            elif matrix_name == 'R0_rect':
                R0 = np.array([float(i) for i in line[1:]]).reshape(3, 3)  # Read from file
                R0 = np.insert(R0, 3, values=0, axis=1)  # Pad with zeros on the right
                R0 = np.insert(R0, 3, values=[0, 0, 0, 1], axis=0)  # Add bottom row
            elif matrix_name == 'P2':
                P2 = np.array([float(i) for i in line[1:]]).reshape(3, 4)
                P2 = np.insert(P2, 3, values=[0, 0, 0, 1], axis=0)  # Add bottom row
    return V2C, R0, P2

def transform(H, pts):
    return H2C(np.dot(H, C2H(pts)))

def C2H(pts):
    return np.insert(pts, 3, values=1, axis=0)

def H2C(pts):
    return pts[:3, :] / pts[3:, :]

def project(P, pts):
    pts = transform(P, pts)
    pts = pts[:2, :] / pts[2, :]
    return pts

def box_filter(pts, box_lim, decorations=None):
    x_range, y_range, z_range = box_lim
    mask = ((pts[0] > x_range[0]) & (pts[0] < x_range[1]) &
            (pts[1] > y_range[0]) & (pts[1] < y_range[1]) &
            (pts[2] > z_range[0]) & (pts[2] < z_range[1]))
    pts = pts[:, mask]
    decorations = decorations[:, mask]
    return pts, decorations

def fov_filter(pts, P, img_size, decorations=None):
    pts_projected = project(P, pts)
    mask = ((pts_projected[0] >= 0) & (pts_projected[0] <= img_size[1]) &
            (pts_projected[1] >= 0) & (pts_projected[1] <= img_size[0]))
    pts = pts[:, mask]
    return pts if decorations is None else pts, decorations[:, mask]

def get_velo(path, calib_path, workspace_lim=((-40, 40), (-1, 2.5), (0, 70)), use_fov_filter=True):
    
    velo = np.fromfile(path, dtype=np.float32).reshape((-1, 4)).T
    pts = velo[0:3]
    reflectance = velo[3:]
    # Transform points from velo coordinates to rectified camera coordinates
    V2C, R0, P2 = get_calib(calib_path)
    pts = transform(np.dot(R0, V2C), pts)
    # Remove points out of workspace
    pts, reflectance = box_filter(pts, workspace_lim, decorations=reflectance)
    # Remove points not projecting onto the image plane
    if use_fov_filter:
        pts, reflectance = fov_filter(pts, P=P2, img_size=(IMG_HEIGHT, IMG_WIDTH), decorations=reflectance)
    return pts, reflectance

