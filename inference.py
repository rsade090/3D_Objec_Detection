from transformers import ResNetConfig, ResNetModel, AutoImageProcessor, TFResNetForImageClassification
import tensorflow as tf
from datasets import load_dataset
import torch 
import Kitti_Process.kittiDataset as KITTI
import config.kitti_config as cnf
from easydict import EasyDict as edict
from torch import optim, nn
from torchvision import models, transforms
from models.featureExtractor import FeatureExtractor
from tqdm import tqdm
import numpy as np

configs = edict()
configs.distributed = False  # For testing
configs.pin_memory = False
configs.num_samples = None
configs.input_size = (370, 1224)
configs.hm_size = (152, 152)
configs.max_objects = 50
configs.num_classes = 3
configs.output_width = 608

configs.dataset_dir = "/home/hooshyarin/Documents/KITTI/"
dataset = KITTI.KittiDataset(configs, mode='test', lidar_aug=None, hflip_prob=0., num_samples=configs.num_samples)

#resnet test with hugging face..................................................................
# image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# model = ResNetModel.from_pretrained("microsoft/resnet-50")
# for idx in range(len(dataset)):
#     bev_map, labels, img_rgb, img_path = dataset.draw_img_with_label(idx)
#     top_view, top_img = dataset.get_BEV(idx)
#     inputs = image_processor(img_rgb, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     last_hidden_states = outputs.last_hidden_state

# model feature extractor .........................................................
model = models.resnet50(pretrained=True)
#model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model,'resnet50')
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

transform = transforms.Compose([
  transforms.ToPILImage(),
  #transforms.CenterCrop(512),
  transforms.Resize((448,448)),
  transforms.ToTensor()                              
])
features = []
for id, data in enumerate(dataset):
    metadata, fov, bev, img = data
    img = transform(img)
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    with torch.no_grad():
        feature = new_model(img)
    features.append(feature.cpu().detach().numpy().reshape(-1))    
    print(id)
features = np.array(features)    
