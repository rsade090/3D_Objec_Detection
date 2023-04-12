# from __future__ import print_function
import os
import sys
import numpy as np
import cv2
from config import kitti_config as cnf

def roty(angle):
    # Rotation about the y-axis.
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def compute_box_3d(dim):
    # dim: 3
    # location: 3
    # ry: 1
    # return: 8 x 3
    R = roty(float(dim[14]))
    h, w, l = float(dim[8]), float(dim[9]), float(dim[10])
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    location = [float(dim[11]), float(dim[12]), float(dim[13])]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return np.array(pts_2d, dtype='int64') #int(pts_2d) 

def draw_box_3d_v2(image, qs, color=(255, 0, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    return image


def draw_box_3d(image, corners, color=(0, 0, 255)):
    ''' Draw 3d bounding box in image
        corners: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    '''

    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                     (int(corners[f[(j + 1) % 4], 0]), int(corners[f[(j + 1) % 4], 1])), color, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                     (int(corners[f[2], 0]), int(corners[f[2], 1])), color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                     (int(corners[f[3], 0]), int(corners[f[3], 1])), color, 1, lineType=cv2.LINE_AA)

    return image


def show_rgb_image_with_boxes(img, labels, calibP2):
    for box_idx, label in enumerate(labels):
        cls_id, location, dim, ry = label[0], label[1:4], label[4:7], label[7]
        if location[2] < 2.0:  # The object is too close to the camera, ignore it during visualization
            continue
        if cls_id < 0:
            continue
        corners_3d = compute_box_3d(dim, ry)
        corners_2d = project_to_image(corners_3d, calibP2)
        img = draw_box_3d(img, corners_2d, color=cnf.colors[int(cls_id)])

    return img

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image
def project_to_image(pts_3d, P):

    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def show_image_with_boxes(img, objects, calib, show3d=True, depth=None):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    cv2.imwrite("mainImg.jpg", img)
    #img3 = np.copy(img)  # for 3d bbox
    #TODO: change the color of boxes
    for obj in objects:
        if "DontCare"== obj[0]:
            continue
        if "Car" == obj[0][0]:
            cv2.rectangle(
            img1,
            (int(float(obj[4][0])), int(float(obj[5][0]))),
            (int(float(obj[6][0])), int(float(obj[7][0]))),
            (0, 255, 0),
            2,
        )
        if "Pedestrian" == obj[0][0]:
            cv2.rectangle(
            img1,
            (int(float(obj[4][0])), int(float(obj[5][0]))),
            (int(float(obj[6][0])), int(float(obj[7][0]))),
            (255, 255, 0),
            2,
        )
        if "Cyclist" == obj[0][0]:
            cv2.rectangle(
            img1,
            (int(float(obj[4][0])), int(float(obj[5][0]))),
            (int(float(obj[6][0])), int(float(obj[7][0]))),
            (0, 255, 255),
            2,
        )
        box3d_pts_2d = compute_box_3d(obj)
        pts = project_to_image(box3d_pts_2d, calib)
        box3d_pts_2d = pts
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        if "Car" == obj[0][0]:
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        elif "Pedestrian" == obj[0][0]:
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        elif "Cyclist" == obj[0][0]:
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))


        # project
        # box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # box3d_pts_32d = utils.box3d_to_rgb_box00(box3d_pts_3d_velo)
        # box3d_pts_32d = calib.project_velo_to_image(box3d_pts_3d_velo)
        # img3 = utils.draw_projected_box3d(img3, box3d_pts_32d)
    # print("img1:", img1.shape)
    cv2.imshow("2dbox", img1)
    # print("img3:",img3.shape)
    # Image.fromarray(img3).show()
    show3d = True
    if show3d:
        # print("img2:",img2.shape)
        cv2.imshow("3dbox", img2)
    if depth is not None:
        cv2.imshow("depth", depth)
    
    return img1, img2

def merge_rgb_to_bev(img_rgb, img_bev, output_width):
    img_rgb_h, img_rgb_w = img_rgb.shape[:2]
    ratio_rgb = output_width / img_rgb_w
    output_rgb_h = int(ratio_rgb * img_rgb_h)
    ret_img_rgb = cv2.resize(img_rgb, (output_width, output_rgb_h))

    img_bev_h, img_bev_w = img_bev.shape[:2]
    ratio_bev = output_width / img_bev_w
    output_bev_h = int(ratio_bev * img_bev_h)

    ret_img_bev = cv2.resize(img_bev, (output_width, output_bev_h))

    out_img = np.zeros((output_rgb_h + output_bev_h, output_width, 3), dtype=np.uint8)
    # Upper: RGB --> BEV
    out_img[:output_rgb_h, ...] = ret_img_rgb
    out_img[output_rgb_h:, ...] = ret_img_bev

    return out_img