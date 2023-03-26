import pandas as pd
import torch
import torchvision
import numpy as np
from PIL import Image
import os
import datetime


checkpoint_dir = 'checkpoints/'
target_checkpoint = 'fullytrained_1_2797.pth'
models_dir = 'models/'


# Define model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 42 # will manually define here to keep code short
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#loading the checkpoint model
checkpoint_path = os.path.join(checkpoint_dir, target_checkpoint)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
start_epoch = checkpoint['epoch']
start_batch = checkpoint['batch']
print(f"Loaded checkpoint {checkpoint_path}")

#saving the model
torch.save(model.state_dict(),os.path.join(models_dir, target_checkpoint))