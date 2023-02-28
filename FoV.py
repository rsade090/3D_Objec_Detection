import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
from PIL import Image



KITTI_DIR = 'D:/Datasets/KITTI/training'
CARS_ONLY = {'Car': ['Car']}
IMG_WIDTH, IMG_HEIGHT = 1242, 375
GT_COLOR = (0.0, 1.0, 0.0)
PRED_COLOR = (1.0, 0.0, 0.0)
BOX_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3], [1, 6], [2, 5]]
class KITTI:
    def __init__(self, ds_dir=KITTI_DIR, class_dict=CARS_ONLY):
        self.ds_dir = ds_dir
        self.img2_dir = os.path.join(ds_dir, 'image_2')
        self.label_dir = os.path.join(ds_dir, 'label_2')
        self.velo_dir = os.path.join(ds_dir, 'velodyne')
        self.calib_dir = os.path.join(ds_dir, 'calib')

        self.class_to_group = {}
        for group, classes in class_dict.items():
            for cls in classes:
                self.class_to_group[cls] = group


def get_range_view( img=None, pts=None, ref=None, P2=None, gt_boxes=None, pred_boxes=None, out_type=None):
    if out_type not in ['depth', 'intensity', 'height']:
        return None

    if pts.shape[0] != 3:
        pts = pts.T
    if ref.shape[0] != 1:
        ref = ref.T

    if img is not None:
        img = np.copy(img)  # Clone
    else:
        img = np.zeros((375, 1242, 1))

    def draw_boxes_2D(boxes, color):
        for box in boxes:
            cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 1)

    def draw_boxes_3D(boxes, color):
        for box in boxes:
            corners = project(P2, box.get_corners()).astype(np.int32)
            for start, end in BOX_CONNECTIONS:
                x1, y1 = corners[:, start]
                x2, y2 = corners[:, end]
                cv2.line(img, (x1, y1), (x2, y2), color, 1)

    # if gt_boxes is not None and len(gt_boxes) > 0:
    #     if isinstance(gt_boxes[0], Box2D):
    #         draw_boxes_2D(gt_boxes, GT_COLOR)
    #     elif isinstance(gt_boxes[0], Box3D):
    #         draw_boxes_3D(gt_boxes, GT_COLOR) 

    if pts is not None:

        pts_projected = project(P2, pts).astype(np.int32).T
        for i in range(pts_projected.shape[0]):
            if out_type == 'depth':
                clr = pts.T[i][2] / 70.
            elif out_type == 'intensity':
                clr = ref.T[i][0]
            elif out_type == 'height':
                clr = 1. - (pts.T[i][1] + 1.) / 4.
            cv2.circle(img, (pts_projected[i][0], pts_projected[i][1]), 4, float(clr), -1)

        # window_sz = 1
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         if np.sum(img[i,j]) <= 0.0:
        #             if i - window_sz >= 0 and j - window_sz >= 0 and i + window_sz <= img.shape[0] and j + window_sz <= img.shape[1]:
        #                 img[i,j] = np.sum(img[i-window_sz:i+window_sz,j-window_sz:j+window_sz], axis=(0,1)) / (window_sz * 2)**2.

        # img = rgb2gray(img)
        # img = np.expand_dims(img, axis=-1)

    return img

def get_calib(path):
    # Read file
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            # Skip if line is empty
            if len(line) == 0:
                continue
            # Load required matrices only
            print('line is ', line)
            matrix_name = line[0][:-1]
            print(line[0])
            print('matrxi is',matrix_name)
    return matrix_name
#calib=get_calib('C:/Users/sadeghianr/Desktop/data/KITTI_original/data_object_calib/training/calib/000000.txt')



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

def get_image(path):
    img = Image.open(path).resize((IMG_WIDTH, IMG_HEIGHT))
    #img = cv2.imread(path)
    #cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return np.asarray(img, dtype=np.float32) / 255.0


if __name__=="__main__":


    cur_idx='C:/Users/sadeghianr/Desktop/data/KITTI_original/data_object_calib/training/calib/000025.txt'
    V2C, R0, P2 = get_calib(cur_idx)
    print("V2C is ", V2C)
    print("R0 is:", R0)
    print("P2 is: ",P2)
    cur_idx='C:/Users/sadeghianr/Desktop/data/KITTI_original/data_object_calib/training/calib/000025.txt'
    vel_dir="C:/Users/sadeghianr/Desktop/data/KITTI_original/training/velodyne/000025.bin"
    im_dir="C:/Users/sadeghianr/Desktop/data/KITTI_original/training/image_2/000025.png"
    pc, ref = get_velo(vel_dir,cur_idx, workspace_lim=((-40, 40), (-1, 3), (0, 70)), use_fov_filter=True)
    print('pc.shape', pc.shape, 'ref.shape', ref.shape)

    depth_img = get_range_view(img=None, pts=pc, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='depth')
    intensity_img = get_range_view(img=None, pts=pc, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='intensity')
    height_img    = get_range_view(img=None, pts=pc, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='height')
    print('depth_img.shape    ', depth_img.shape)
    print('intensity_img.shape', intensity_img.shape)
    print('height_img.shape   ', height_img.shape)

    img = get_image(im_dir)
    print('img.shape', img.shape)
    plt.figure(figsize = (20,10))
    #img.show()
    plt.imshow(img)
    #plt.show()
    
    plt.figure(figsize = (20,8))
    plt.imshow(np.squeeze(depth_img))
    plt.figure(figsize = (20,8))
    plt.imshow(np.squeeze(intensity_img))
    plt.figure(figsize = (20,8))
    plt.imshow(np.squeeze(height_img))
    plt.show()