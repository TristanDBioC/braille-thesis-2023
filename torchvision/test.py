import torch
import torchvision
import numpy as np
from PIL import Image
import os
import cv2


# defining paths and directories
models_dir = 'models/'
model_file = 'fullytrained_1_2797.pth'
img_path = 'test_images/1ABC.jpg'




model_pth = os.path.join(models_dir, model_file)

# define what the model is
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 42 # will manually define here to keep code short
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Define anchor sizes
anchor_gen = AnchorGenerator(sizes=((4, 8, 16, 32, 64),),
                                   aspect_ratios=((0.5, 1.0, 2.0), * 5))
model.rpn.anchor_generator = anchor_gen


# loading the model
model.load_state_dict(torch.load(model_pth))
model.eval()

#initializing image
img = Image.open(img_path)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(1)
])
img = transform(img)


#prediction part
with torch.no_grad():
    pred = model([img])

boxes, labels, scores = pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']

num = torch.argwhere(scores > 0.075).shape[0]

igg = cv2.imread(img_path)
for i in range(num):
    x1, y1, x2, y2 = boxes[i].numpy().astype(int)
    print(x1, y1, x2, y2)
    igg = cv2.rectangle(igg, (x1, y1), (x2, y2), (255, 0, 0))


cv2.imshow('preview', igg)
cv2.waitKey(0)