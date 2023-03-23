import pandas as pd
import torch
import torchvision
import numpy as np
from PIL import Image
import os
import datetime


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

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Define data loaders for training and testing sets
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = len(dataset.labels) + 1 # +1 for background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move model to device
model.to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# Define checkpointing interval
checkpoint_interval = 100

# Define directory to save checkpoints
checkpoint_dir = 'checkpoints/'

# Load the latest saved checkpoint if available
start_epoch = 0
start_batch = 0
latest_checkpoint = max([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')], default=None)
if latest_checkpoint is not None:
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint['batch']
    print(f"Loaded checkpoint {checkpoint_path}, starting from epoch {start_epoch+1} and batch {start_batch+1}")


# Train the model
if __name__ == '__main__':
    print('Number of classess: ',len(dataset.label_map))
    print('Device:', device)
    print('training start time: ', datetime.datetime.now())
    num_epochs = 1
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (images, targets) in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                print(f'Skipping epoch {epoch+1} batch {batch_idx+1}')
                continue  # Skip batches that were already trained
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()

            if (batch_idx + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch+1}_{batch_idx+1}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'batch': batch_idx
                }, checkpoint_path)
                print(f'Saved checkpoint {checkpoint_path}')
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {losses.item():.4f}")
    print(f"Finished training: {datetime.datetime.now()}")
