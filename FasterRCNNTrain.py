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
from Kitti_Process.kitti_bev_utils import draw_box3d_on_top
from dataAugmentation import DataAugmentation

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
configs.hm_size = (152, 152)  #RRRR### marboot be config
configs.max_objects = 50
configs.num_classes = 3
configs.dataset_dir = "/home/sadeghianr/Desktop/Datasets/Kitti/"
configs.imageSize =  (375,1242)#(256,256) # (375,1242)(192, 640)    ##RRRR###
train_set = KittiDataset(configs, mode='train', lidar_aug=None, hflip_prob=0.)
test_set = KittiDataset(configs, mode='train', lidar_aug=None, hflip_prob=0.)
print('number of train samples: ', len(train_set))
print('number of test samples: ', len(train_set))

dataloader_train = DataLoader(train_set, batch_size=8, shuffle=True,collate_fn=train_set.collate_fn, num_workers=8, pin_memory=True)
dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False,collate_fn=train_set.collate_fn, num_workers=8, pin_memory=True)

# create anchor boxes
anc_scales = [2, 4, 6]
anc_ratios = [0.5, 1, 1.5]
n_anc_boxes = len(anc_scales) * len(anc_ratios) # number of anchor boxes for each anchor point
out_c, out_h, out_w = 256, 16, 16 #256, 12, 40 #2048, 15, 20      ##RRRR#### output transformer niyaz baraye twostage detector
anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(out_h, out_w))   ##RRR###
anc_base = gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, (out_h, out_w))

width_scale_factor = configs.imageSize[1] // out_w
height_scale_factor = configs.imageSize[0] // out_h
out_size = (out_h, out_w)
name2idx = kittiCnf.CLASS_NAME_TO_ID
idx2name = {v:k for k, v in name2idx.items()}
n_classes = len(name2idx) #-1 # exclude pad idx
roi_size = (2, 2)     ###RRRR### vase pooling


detector = TwoStageDetector(configs.imageSize, out_size, out_c, n_classes, roi_size)
detector.to(args.device)
optimizer = optim.Adam(detector.parameters(), lr=0.0001)
#transform = transforms.Compose([
  #transforms.Resize((216,640)),                         
  #transforms.Resize((480,640)),
  #transforms.RandomCrop(256)  ####RRRR####                       
#])
transform = transforms.Compose([transforms.Resize((256,256)),])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
param_count = count_parameters(detector)
# number of trainable parameters -> 35848757


BOX_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3]] ##RRR### connection corner of cube
def draw_rect(img, corners, filename):
  img3 = img[0] # vase namayesh dim 0 ke nehsndahandeye batch ast bardashte shode
  for i in range(corners.shape[1]):
    minx = min(corners[0][i][:,0]).numpy()#corners [batch][box][coordinate]
    miny = min(corners[0][i][:,1]).numpy()
    maxx = max(corners[0][i][:,0]).numpy()
    maxy = max(corners[0][i][:,1]).numpy()
    img3 = cv2.rectangle(img[0].cpu().numpy(), (int(minx), int(miny)), (int(maxx), int(maxy)), (255,0,0),1)# chap-bala(min), rast-payin(max) 
    imt = img[0]#vase namayesh dim 0 ke nehsndahandeye batch ast bardashte shode
    imt = imt[int(miny): int(maxy), int(minx): int(maxx)]#crop ax vase namayesh(vase debug)
    print()
    #cv2.putText(img3, classes_pred_1[i], (int(minx), int(miny)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
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

dataAug = DataAugmentation()

TrainMode = True
#detector.load_state_dict(torch.load("/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/model_weights/crop256_justRandomCroppedimage/model390.pt"))
if TrainMode:
  epochs = 800
  loss_list = []
  for i in range(epochs):
    total_loss = 0
    count = 0
    sample = 0
    for data in tqdm(dataloader_train):        
        img, bev, fov, targetBox, targetLabel,x = data
        img, bev, fov, targetBox, targetLabel,x = dataAug.randPosCrop(img,targetBox,targetLabel, x, bev, fov)
        sample += img.shape[0]
        imgs =  torch.permute(img, (0,3, 1, 2)).to(args.device, dtype=torch.float32)

        ### if it is not cropped pr resized to (256*256) following command should be uncommented.##

        #imgs =  transform(torch.permute(img, (0,3, 1, 2))).to(args.device, dtype=torch.float32)


        bevs = torch.permute(bev, (0,3, 1, 2)).to(args.device, dtype=torch.float32)


        targetB = [v.to(args.device, dtype=torch.float32) for v in targetBox]
        targetL = [t.to(args.device, dtype=torch.int64) for t in targetLabel]
        detector.train()
        loss = detector(imgs, bevs, targetB, targetL)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    writer.add_scalar("Loss/train", total_loss/sample, i) #ezafe kardan data be tensorboard
    #save model
    if i % 10==0:
      torch.save(detector.state_dict(), "/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/model_weights/crop256_justRandomCroppedimage/model"+str(i+400)+".pt")
    loss_list.append(total_loss/len(dataloader_train))
  writer.flush()  
  

# Test and inference the model
# load the model
detector.load_state_dict(torch.load("/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/model_weights/crop256_justRandomCroppedimage/model390.pt"))
testMode = False
count = 0
for data in tqdm(dataloader_test):

  img, bev, fov, targetBox, targetLabel = data
  img, bev, fov, targetBox, targetLabel = dataAug.randPosCrop(img,targetBox,targetLabel, bev, fov)
  if count < 50 :
    im = img.clone()
    draw_rect(im, targetBox, "/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/imageResults/crop256_justRandomCroppedimage/beforeRect"+str(count))
    im = img.clone()
    draw_Cube(im, targetBox, "/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/imageResults/crop256_justRandomCroppedimage/beforeCube"+str(count))
  else:
     print()
  imgs = (torch.permute(img, (0,3, 1, 2))).to(args.device, dtype=torch.float32)
  bevs = torch.permute(bev, (0,3, 1, 2)).to(args.device, dtype=torch.float32)#(transform
  targetB = [v.to(args.device, dtype=torch.float32) for v in targetBox]
  targetL = [t.to(args.device, dtype=torch.int64) for t in targetLabel]
  detector.eval()
  proposals_final, conf_scores_final, classes_final = detector.inference(imgs, bevs, conf_thresh=0.99, nms_thresh=0.01) 
  # project proposals to the image space
  proposals_final = pad_sequence(proposals_final, batch_first=True, padding_value=-1)
  prop_proj_1 = project_bboxes(proposals_final, width_scale_factor, height_scale_factor, mode='a2p')
  classes_pred_1 = [idx2name[cls] for cls in classes_final[0].tolist()]
  # prop_proj_1[0][0][0] = int(float(prop_proj_1[0][0][0])) / (img[0].shape[1]/ 1242)
  # prop_proj_1[0][0][1] = int(float(prop_proj_1[0][0][1])) / (img[0].shape[1]/ 1242)
  # prop_proj_1[0][0][2] = int(float(prop_proj_1[0][0][2])) / (img[0].shape[0]/ 375)
  # prop_proj_1[0][0][3] = int(float(prop_proj_1[0][0][3])) / (img[0].shape[0]/ 375)
  #imm = img[0]
  # bboxes = ops.box_convert(prop_proj_1[0], in_fmt='xyxy', out_fmt='xywh')
  # prop_proj_1[0][0][0] = bboxes[0][0]
  # prop_proj_1[0][0][1] = bboxes[0][1]
  # prop_proj_1[0][0][2] = bboxes[0][0] + bboxes[0][2]
  # prop_proj_1[0][0][3] = bboxes[0][1] + bboxes[0][3]
  #img3 = cv2.rectangle(img[0].cpu().numpy(), (int(bboxes[0][0]), int(bboxes[0][1])), (int(bboxes[0][2]), int(bboxes[0][3])), (255,0,0),1)
  img4 = (img[0].clone()).cpu().numpy()
  for i in range(prop_proj_1[0].shape[0]):
    img4 = cv2.rectangle(img[0].cpu().numpy(), (int(prop_proj_1[0][i][0]), int(prop_proj_1[0][i][1])), (int(prop_proj_1[0][i][2]), int(prop_proj_1[0][i][3])), (0,255,0),1)
    cv2.putText(img4, classes_pred_1[i], (int(prop_proj_1[0][i][0]), int(prop_proj_1[0][i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36,255,12), 1)
    if count < 50 :
      cv2.imwrite("/home/sadeghianr/Desktop/Codes/3D_Objec_Detection/imageResults/crop256_justRandomCroppedimage/result"+str(count)+".jpg", img4)  
  count+=1

  # nrows, ncols = (2, 2)
  # fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
  # fig, axes = display_img(imgs, fig, axes)
  # fig, _ = display_bbox(prop_proj_1[0], fig, axes[0], classes=classes_pred_1)

