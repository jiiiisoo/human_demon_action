# LIBERO Training Data Extraction Guide

This guide explains how to extract and use LIBERO demonstration data for training your models.

## Quick Start

### 1. Extract Training Data

Extract agentview RGB images and actions from an HDF5 dataset:

```bash
python extract_training_data.py \
    --dataset /mnt/data/libero/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 \
    --output ./training_data
```

**Options:**
- `--dataset`: Path to your HDF5 dataset file (required)
- `--output`: Output directory (default: `./extracted_data`)
- `--png`: Save as individual PNG files instead of numpy arrays
- `--filter-key`: Use a specific filter key from the dataset

### 2. What Gets Extracted

The extraction creates:

```
training_data/
├── all_images.npy      # All images concatenated, shape: (N, 128, 128, 3)
├── all_actions.npy     # All actions concatenated, shape: (N, 7)
└── metadata.json       # Dataset information and demo boundaries
```

**Metadata includes:**
- Language instruction for the task
- Number of demonstrations and total frames
- Frame indices for each demonstration
- Image shape and action dimension

### 3. Load Data for Training

#### Using the Example DataLoader

```bash
python example_training_dataloader.py
```

#### In Your Own Code

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Load the data
images = np.load('training_data/all_images.npy')  # Shape: (N, 128, 128, 3)
actions = np.load('training_data/all_actions.npy')  # Shape: (N, 7)

# Or use the provided PyTorch Dataset
from example_training_dataloader import LIBERODataset

dataset = LIBERODataset('training_data')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    images = batch['image']    # Shape: (batch_size, 3, 128, 128)
    actions = batch['action']  # Shape: (batch_size, 7)
    # Your training code here
```

## Data Format

### Images
- **Shape**: `(N, 128, 128, 3)` where N is total number of frames
- **Type**: uint8 (0-255)
- **Source**: `agentview_rgb` camera
- **Format**: RGB images

### Actions
- **Shape**: `(N, 7)` where N is total number of frames
- **Type**: float64
- **Range**: [-1.0, 1.0]
- **Dimensions**: 
  - 0-2: End-effector position delta (x, y, z)
  - 3-5: End-effector orientation delta (roll, pitch, yaw)
  - 6: Gripper action (-1 = close, 1 = open)

## Advanced Usage

### Extract from Multiple Datasets

```bash
# Extract libero_10 task
python extract_training_data.py \
    --dataset /mnt/data/libero/libero_10/task1_demo.hdf5 \
    --output ./data/task1

# Extract libero_90 task
python extract_training_data.py \
    --dataset /mnt/data/libero/libero_90/task2_demo.hdf5 \
    --output ./data/task2
```

### Save as PNG Files (for visualization)

```bash
python extract_training_data.py \
    --dataset /path/to/dataset.hdf5 \
    --output ./data_png \
    --png
```

This creates:
```
data_png/
├── demo_0/
│   ├── frame_00000.png
│   ├── frame_00001.png
│   ├── ...
│   └── actions.npy
├── demo_1/
│   └── ...
└── metadata.json
```

### Access Individual Demonstrations

```python
import json
import numpy as np

# Load metadata
with open('training_data/metadata.json', 'r') as f:
    metadata = json.load(f)

# Load all data
images = np.load('training_data/all_images.npy')
actions = np.load('training_data/all_actions.npy')

# Get data for a specific demonstration
demo = metadata['demo_info'][0]  # First demo
demo_images = images[demo['start_idx']:demo['end_idx']]
demo_actions = actions[demo['start_idx']:demo['end_idx']]

print(f"Demo: {demo['demo_name']}")
print(f"Frames: {demo['num_frames']}")
print(f"Images shape: {demo_images.shape}")
print(f"Actions shape: {demo_actions.shape}")
```

## View Dataset Information

To inspect a dataset without extracting:

```bash
cd libero/libero/utils
python dataset_utils.py
```

Or in Python:

```python
from libero.libero.utils.dataset_utils import get_dataset_info

get_dataset_info('/path/to/dataset.hdf5', verbose=True)
```

## Tips for Training

1. **Memory Efficiency**: The numpy format is more memory-efficient than PNG files
2. **Normalization**: Remember to normalize images to [0, 1] or [-1, 1] for training
3. **Data Augmentation**: Consider adding augmentation (random crops, color jitter, etc.)
4. **Multi-task Learning**: Extract multiple tasks and train on all of them
5. **Trajectory Boundaries**: Use `demo_info` to respect demonstration boundaries

## Example Training Script Structure

```python
import torch
import torch.nn as nn
from example_training_dataloader import LIBERODataset

# Load dataset
dataset = LIBERODataset('./training_data')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define your model
class MyRobotPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model architecture
        
    def forward(self, image):
        # Predict action from image
        return action

model = MyRobotPolicy()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch['image'].cuda()
        actions = batch['action'].cuda()
        
        # Forward pass
        predicted_actions = model(images)
        loss = criterion(predicted_actions, actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Troubleshooting

**Issue**: Out of memory when loading data
- **Solution**: Use PyTorch's memory-mapped loading or load data in chunks

**Issue**: Images look wrong
- **Solution**: Check if you need to normalize or convert RGB/BGR

**Issue**: Actions seem incorrect
- **Solution**: Remember actions are deltas in range [-1, 1], not absolute positions

## Files Overview

- `extract_training_data.py`: Main extraction script
- `example_training_dataloader.py`: PyTorch DataLoader example
- `libero/libero/utils/dataset_utils.py`: Core extraction utilities




























