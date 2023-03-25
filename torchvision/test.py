import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Get unique labels and assign an integer index to each label
        self.labels = self.annotations['label'].unique().tolist()
        self.label_map = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx, 0]
        label = self.annotations.iloc[idx, 1]
        bbox = self.annotations.iloc[idx, 2:].values.astype('int32')

        # Convert label to integer index using label_map
        label = self.label_map[label]

        # Load image
        img_path = os.path.join(self.root_dir, img_path)
        img = Image.open(img_path).convert("RGB")

        # Apply transform if specified
        if self.transform:
            img = self.transform(img)

        # Create target dictionary
        target = {}
        target['boxes'] = torch.tensor(np.array([bbox]))
        target['labels'] = torch.tensor([label])

        return img, target


# Define transformations for the input images
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    torchvision.transforms.Grayscale(1)
])

# Create custom dataset
dataset = CustomDataset(csv_file='labels.csv', root_dir='', transform=transform)

# Define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = len(dataset.labels) + 1 # +1 for background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move model to device
model.to(device)

# Load the saved checkpoint
checkpoint_path = 'checkpoints/checkpoint_1_10.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Define data loader for validation set
val_dataset = CustomDataset(csv_file='val_labels.csv', root_dir='', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# Define criterion for calculating the validation loss
criterion = torch.nn.L1Loss(reduction='mean')

# Calculate validation loss
val_loss = 0.0
with torch.no_grad():
    for images, targets in val_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Calculate the output of the model
        output = model(images, targets)
        # Flatten the output bounding box coordinates and targets for loss calculation
        output_boxes = torch.flatten(output[0]['boxes'])
        target_boxes = torch.flatten(targets[0]['boxes'])
        # Calculate the validation loss
        val_loss += criterion(output_boxes, target_boxes)

