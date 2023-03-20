import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from config.Transfuse_config import GlobalConfig
from models.TransfuserModel import TransFuser
from easydict import EasyDict as edict
from Kitti_Process.kittiDataset import KittiDataset
from torchvision import models, transforms
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from models.ObjectDetModel import ObjDet
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='transfuser', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=101, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)

writer = SummaryWriter(log_dir=args.logdir)
transform = transforms.Compose([
  #transforms.ToPILImage(),
  #transforms.CenterCrop(512),
  transforms.Resize((256,256)),
  #transforms.ToTensor()                              
])


def create_model(num_classes, backboneNew):

    conv1 = torchvision.models.resnet18(pretrained=True).conv1
    bn1 = torchvision.models.resnet18(pretrained=True).bn1
    resnet18_relu = torchvision.models.resnet18(pretrained=True).relu
    resnet18_max_pool = torchvision.models.resnet18(pretrained=True).maxpool
    layer1 = torchvision.models.resnet18(pretrained=True).layer1
    layer2 = torchvision.models.resnet18(pretrained=True).layer2
    layer3 = torchvision.models.resnet18(pretrained=True).layer3
    layer4 = torchvision.models.resnet18(pretrained=True).layer4

    backbone = nn.Sequential(
        conv1, bn1, resnet18_relu, resnet18_max_pool, 
        layer1, layer2, layer3, layer4
    )
    backbone.out_channels = 512

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Pascal VOC Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=21,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    # Load the PASCAL VOC pretrianed weights.
    # print('Loading PASCAL VOC pretrained weights...')
    # checkpoint = torch.load('../input/pretrained_voc_weights/last_model_state.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])

    # Replace the Faster RCNN head with required class number. Final model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    print(roi_pooler)
    return model


class Engine(object):

    def __init__(self,  cur_epoch=0, cur_iter=0):
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        model.to(args.device).train()
        for data in tqdm(dataloader_train):
            for p in model.parameters():
                p.grad = None

            metadata, fov, bev, img, target = data
            imgs = torch.permute(img, (0,3, 1, 2))
            imgs = transform(imgs)   
            imgs = imgs.to(args.device, dtype=torch.float32)
            bevs = (transform(torch.permute(bev[0], (0,3, 1, 2)))).to(args.device, dtype=torch.float32)
            target['boxes'] = [v.to(args.device, dtype=torch.float32) for v in target['boxes']]
            target['labels'] = [t.to(args.device, dtype=torch.int64) for t in target['labels']]
            input = [[imgs], [bevs]]
            output = model(input)
            pred_scores, pred_boxes = output
            # gt_boxes = [torch.stack(target['boxes'][i].reshape(-1,24), dim=1).to(args.device, dtype=torch.float32) for i in range(0, len(target['boxes']))]
            # gt_boxes = torch.stack(gt_boxes, dim=1).to(args.device, dtype=torch.float32)
            loss = F.l1_loss(pred_boxes, target['boxes'][0].reshape(-1,24), reduction='none').mean()
            loss.backward()
            loss_epoch += float(loss.item())
            num_batches += 1
            optimizer.step()
            writer.add_scalar('train_loss', loss.item(), self.cur_iter)
            self.cur_iter += 1
               
        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self):
        model.eval()
        with torch.no_grad():	
            num_batches = 0
            wp_epoch = 0.
            for batch_num, data in enumerate(tqdm(dataloader_val), 0):
                fronts_in = data['fronts']
                lefts_in = data['lefts']
                rights_in = data['rights']
                rears_in = data['rears']
                lidars_in = data['lidars']
                fronts = []
                lefts = []
                rights = []
                rears = []
                lidars = []
                for i in range(config.seq_len):
                    fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
                    if not config.ignore_sides:
                        lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
                        rights.append(rights_in[i].to(args.device, dtype=torch.float32))
                    if not config.ignore_rear:
                        rears.append(rears_in[i].to(args.device, dtype=torch.float32))
                    lidars.append(lidars_in[i].to(args.device, dtype=torch.float32))

                # driving labels
                command = data['command'].to(args.device)
                gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)
                gt_steer = data['steer'].to(args.device, dtype=torch.float32)
                gt_throttle = data['throttle'].to(args.device, dtype=torch.float32)
                gt_brake = data['brake'].to(args.device, dtype=torch.float32)

                # target point
                target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

                pred_wp = model(fronts+lefts+rights+rears, lidars, target_point, gt_velocity)

                gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
                gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
                wp_epoch += float(F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean())
                num_batches += 1

            wp_loss = wp_epoch / float(num_batches)
            #tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')
            writer.add_scalar('val_loss', wp_loss, self.cur_epoch)            
            self.val_loss.append(wp_loss)

    def save(self):
        save_best = False
        if self.val_loss[-1] <= self.bestval:
            self.bestval = self.val_loss[-1]
            self.bestval_epoch = self.cur_epoch
            save_best = True        
        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
            'bestval': self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }

        # Save ckpt for every epoch
        torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))
        # Save the recent model/optimizer states
        torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))
        # Log other data corresponding to the recent model
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))
        tqdm.write('====== Saved recent model ======>')        
        if save_best:
            torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
            tqdm.write('====== Overwrote best model ======>')

# Config
config = GlobalConfig()
configs = edict()
configs.distributed = False
configs.pin_memory = False
configs.num_samples = None
configs.input_size = (370, 1224)
configs.hm_size = (152, 152)
configs.max_objects = 50
configs.num_classes = 3
configs.output_width = 608
configs.dataset_dir = "/home/hooshyarin/Documents/KITTI/"

train_set = KittiDataset(configs, mode='val', lidar_aug=None, hflip_prob=0., num_samples=configs.num_samples)
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

val_set = KittiDataset(configs, mode='val', lidar_aug=None, hflip_prob=0., num_samples=configs.num_samples)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model
#model = TransFuser(config, args.device)
#fastRCNN with custom backbone

# model = create_model(1,None)
transfModel = TransFuser(config, args.device)
model = ObjDet(transfModel, 1)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Engine()


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	print('Loading checkpoint from ' + args.logdir)
	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
		log_table = json.load(f)

	# Load variables
	trainer.cur_epoch = log_table['epoch']
	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
	trainer.bestval = log_table['bestval']
	trainer.train_loss = log_table['train_loss']
	trainer.val_loss = log_table['val_loss']

	# Load checkpoint
	model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
	optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))

# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)

for epoch in range(trainer.cur_epoch, args.epochs): 
	trainer.train()
	if epoch % args.val_every == 0: 
		trainer.validate()
		trainer.save()