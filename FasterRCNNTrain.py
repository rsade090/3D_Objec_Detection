import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from utils import *
from utils.visualization_utils import *
from models.fasterRCNN import *
import os
import argparse
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from Kitti_Process.kittiDataset import KittiDataset
from easydict import EasyDict as edict
from tqdm import tqdm
import config.kitti_config as kittiCnf
import cv2 as cv
parser = argparse.ArgumentParser()
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
writer = SummaryWriter()

parser.add_argument('--id', type=str, default='transfuser', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=101, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

args = parser.parse_args()
configs = edict()
configs.hm_size = (152, 152)
configs.max_objects = 50
configs.num_classes = 3
configs.dataset_dir = "/home/sadeghianr/Desktop/Datasets/Kitti/"


train_set = KittiDataset(configs, mode='train', lidar_aug=None, hflip_prob=0.)
dataloader_train = DataLoader(train_set, batch_size=16, shuffle=True,collate_fn=train_set.collate_fn,num_workers=16,  pin_memory=True) #,

test_set = KittiDataset(configs, mode='val', lidar_aug=None, hflip_prob=0.)
dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False,collate_fn=train_set.collate_fn, pin_memory=True)

# create anchor boxes
anc_scales = [2, 4, 6]
anc_ratios = [0.5, 1, 1.5]
n_anc_boxes = len(anc_scales) * len(anc_ratios) # number of anchor boxes for each anchor point
out_c, out_h, out_w = 256, 16, 16 #2048, 15, 20
anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(out_h, out_w))
anc_base = gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, (out_h, out_w))

img_size = (256, 256)
width_scale_factor = img_size[1] // out_w
height_scale_factor = img_size[0] // out_h
out_size = (out_h, out_w)
name2idx = kittiCnf.CLASS_NAME_TO_ID
idx2name = {v:k for k, v in name2idx.items()}
n_classes = len(name2idx) #-1 # exclude pad idx
roi_size = (2, 2)


detector = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size)
detector.to(args.device)
optimizer = optim.Adam(detector.parameters(), lr=0.0001)
transform = transforms.Compose([
  transforms.Resize((256,256)),                         
  #transforms.Resize((480,640)),                         
])
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
param_count = count_parameters(detector)
# number of trainable parameters -> 35848757
def display_img(img_data, fig, axes):
    for i, img in enumerate(img_data):
        if type(img) == torch.Tensor:
          img = img.permute(1, 2, 0)
        axes[i].imshow(img)
    
    return fig, axes

def display_bbox(bboxes, fig, ax, classes=None, in_format='xyxy', color='y', line_width=3):
    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if classes:
        assert len(bboxes) == len(classes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')
    c = 0
    for box in bboxes:
        x, y, w, h = box.numpy()
        # display bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # display category
        if classes:
            if classes[c] == 'pad':
                continue
            ax.text(x + 5, y + 20, classes[c], bbox=dict(facecolor='yellow', alpha=0.5))
        c += 1  
    return fig, ax

BOX_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3]]
def draw_rect(img, corners, filename):
  img3 = img[0]
  for i in range(corners.shape[1]):
    minx = min(corners[0][i][:,0]).numpy()
    miny = min(corners[0][i][:,1]).numpy()
    maxx = max(corners[0][i][:,0]).numpy()
    maxy = max(corners[0][i][:,1]).numpy()
    img3 = cv2.rectangle(img[0].cpu().numpy(), (int(minx), int(miny)), (int(maxx), int(maxy)), (255,0,0),1)
  cv2.imwrite(filename+'.jpg', img3)       
  return

def draw_Cube(img, corners,filename):
  for i in range(corners.shape[1]):
    for start, end in BOX_CONNECTIONS:
      x1, y1 = corners[0][i][start,:]
      x2, y2 = corners[0][i][end,:]      
      im = cv2.line(img[0].cpu().numpy(), (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)  
      cv2.imwrite(filename+'.jpg',im)
  return      

TrainMode = False
detector.load_state_dict(torch.load("/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/model_weights/2d_256_256_train_b16_5cls/model240.pt"))
if TrainMode:
  epochs = 800
  loss_list = []
  for i in range(epochs):
    total_loss = 0
    count = 0
    for data in tqdm(dataloader_train):        
        img, bev, fov, targetBox, targetLabel = data
        imgs =  ((torch.permute(img, (0,3, 1, 2)))).to(args.device, dtype=torch.float32)
        bevs = ((torch.permute(bev, (0,3, 1, 2)))).to(args.device, dtype=torch.float32) #transform(
        targetB = [v.to(args.device, dtype=torch.float32) for v in targetBox]
        targetL = [t.to(args.device, dtype=torch.int64) for t in targetLabel]
        detector.train()
        loss = detector(imgs, bevs, targetB, targetL)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    writer.add_scalar("Loss/train", total_loss/len(dataloader_train), i)
    #save model
    if i % 10==0:
      torch.save(detector.state_dict(), "/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/model_weights/2d_256_256_train_b16_5cls/model"+str(i+140)+".pt")
    loss_list.append(total_loss/len(dataloader_train))
  writer.flush()  
  print()

# Test and inference the model
# load the model
detector.load_state_dict(torch.load("/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/model_weights/2d_256_256_train_b16_5cls/model750.pt"))
testMode = True
count = 0
for data in tqdm(dataloader_test):

  img, bev, fov, targetBox, targetLabel = data
  
  if count < 50 :
    im = img.clone()
    draw_rect(im, targetBox, "/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/imageResults/imageResults_5cls/beforeRect"+str(count))
    im = img.clone()
    draw_Cube(im, targetBox, "/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/imageResults/imageResults_5cls/beforeCube"+str(count))
  else:
     print()
  imgs = (transform(torch.permute(img, (0,3, 1, 2)))).to(args.device, dtype=torch.float32)
  bevs = torch.permute(bev, (0,3, 1, 2)).to(args.device, dtype=torch.float32)#(transform
  targetB = [v.to(args.device, dtype=torch.float32) for v in targetBox]
  targetL = [t.to(args.device, dtype=torch.int64) for t in targetLabel]
  detector.eval()
  proposals_final, conf_scores_final, classes_final = detector.inference(imgs, bevs, conf_thresh=0.99, nms_thresh=0.05) 
  # project proposals to the image space
  proposals_final = pad_sequence(proposals_final, batch_first=True, padding_value=-1)
  prop_proj_1 = project_bboxes(proposals_final, width_scale_factor, height_scale_factor, mode='a2p')
  
  #get classes
  classes_pred_1 = [idx2name[cls] for cls in classes_final[0].tolist()]
  # prop_proj_1[0][0][0] = int(float(prop_proj_1[0][0][0])) / (img[0].shape[1]/ 1242)
  # prop_proj_1[0][0][1] = int(float(prop_proj_1[0][0][1])) / (img[0].shape[1]/ 1242)
  # prop_proj_1[0][0][2] = int(float(prop_proj_1[0][0][2])) / (img[0].shape[0]/ 375)
  # prop_proj_1[0][0][3] = int(float(prop_proj_1[0][0][3])) / (img[0].shape[0]/ 375)
  #imm = img[0]
  #bboxes = ops.box_convert(prop_proj_1[0], in_fmt='xywh', out_fmt='xyxy')
  #img3 = cv2.rectangle(img[0].cpu().numpy(), (int(bboxes[0][0]), int(bboxes[0][1])), (int(bboxes[0][2]), int(bboxes[0][3])), (255,0,0),1)
  for i in range(prop_proj_1[0].shape[0]):
    img4 = cv2.rectangle(img[0].cpu().numpy(), (int(prop_proj_1[0][i][0]), int(prop_proj_1[0][i][1])), (int(prop_proj_1[0][i][2]), int(prop_proj_1[0][i][3])), (0,255,0),1)
    cv2.putText(img4, classes_pred_1[i], (int(prop_proj_1[0][i][0]), int(prop_proj_1[0][i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36,255,12), 1)
    if count < 50 :
      cv2.imwrite("/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/imageResults/imageResults_5cls/result"+str(count)+".jpg", img4)  
  count+=1

  # nrows, ncols = (2, 2)
  # fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
  # fig, axes = display_img(imgs, fig, axes)
  # fig, _ = display_bbox(prop_proj_1[0], fig, axes[0], classes=classes_pred_1)

