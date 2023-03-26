import pandas as pd
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
anchor_gen = AnchorGenerator(sizes=((4, 8, 16, 32, 64),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

model.rpn.anchor_generator = anchor_gen

print(model)