import torch
from torch import optim, nn
from torchvision import models, transforms

class FeatureExtractor(nn.Module):
  
  def __init__(self, model, name):
    super(FeatureExtractor, self).__init__()
    self.name = name
    self.modeules = None
    if name == 'resnet50':
      modules=list(model.children())[:-1]
      self.modeules =nn.Sequential(*modules)
    else:  
      self.features = list(model.features)
      self.features = nn.Sequential(*self.features)
      self.pooling = model.avgpool
      self.flatten = nn.Flatten()
      self.fc = model.classifier[0]
  
  def forward(self, x):
    if self.name == 'resnet50':
      out = self.modeules(x)
    else:  
      out = self.features(x)
      out = self.pooling(out)
      out = self.flatten(out)
      out = self.fc(out) 
    return out 


