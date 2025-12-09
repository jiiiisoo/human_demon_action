#!/bin/bash

# LIBERO HDF5 DataLoader Quick Start Script

echo "=========================================="
echo "LIBERO HDF5 DataLoader Quick Start"
echo "=========================================="
echo ""

# Check if data directory exists
BASE_DIR="/mnt/data/libero"
if [ ! -d "$BASE_DIR" ]; then
    echo "❌ Error: Base directory not found: $BASE_DIR"
    echo "Please update BASE_DIR in this script to your actual data location."
    exit 1
fi

echo "✓ Base directory found: $BASE_DIR"
echo ""

# Step 1: Create dataset index
echo "Step 1: Creating dataset index for all suites..."
echo "----------------------------------------"
python create_dataset_index.py \
    --base-dir $BASE_DIR \
    --output ./libero_all_index.json

if [ $? -ne 0 ]; then
    echo "❌ Failed to create index"
    exit 1
fi

echo ""
echo "✓ Index created: ./libero_all_index.json"
echo ""

# Step 2: Test dataloader
echo "Step 2: Testing DataLoader..."
echo "----------------------------------------"
python example_hdf5_dataloader.py \
    --index ./libero_all_index.json \
    --batch-size 32 \
    --num-workers 4

if [ $? -ne 0 ]; then
    echo "❌ DataLoader test failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "You can now use the dataset in your training code:"
echo ""
echo "  from example_hdf5_dataloader import LIBEROHdf5Dataset"
echo "  dataset = LIBEROHdf5Dataset('./libero_10_index.json')"
echo "  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)"
echo ""
echo "See README_HDF5_DATALOADER.md for more details."
echo ""

