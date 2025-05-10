#!/bin/bash

# Script to run the pruner with different percentages
# Creates pruned models at 10%, 20%, and 30% pruning levels

# Default values (can be overridden by environment variables)
CSV_PATH=${CSV_PATH:-"data/stats/filter_stats.csv"} #path to csv
MODEL_PATH=${MODEL_PATH:-"data/models/resnet18.pth"} #path to model original
DATA_DIR=${DATA_DIR:-"data/10k-imagenet"} #path to retrain data
OUTPUT_DIR=${OUTPUT_DIR:-"pruned_outputs"}
EPOCHS=${EPOCHS:-5} #specify epochs
BATCH_SIZE=${BATCH_SIZE:-64}

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Install torch_pruning if not already installed
pip install torch-pruning

# Run pruner with different percentages
echo "Running pruner with 10% pruning..."
python pruner/finetune.py \
  --percentage 0.1 \
  --epochs ${EPOCHS} \
  --csv ${CSV_PATH} \
  --model ${MODEL_PATH} \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE}

echo "Running pruner with 20% pruning..."
python pruner/finetune.py \
  --percentage 0.2 \
  --epochs ${EPOCHS} \
  --csv ${CSV_PATH} \
  --model ${MODEL_PATH} \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE}

echo "Running pruner with 30% pruning..."
python pruner/finetune.py \
  --percentage 0.3 \
  --epochs ${EPOCHS} \
  --csv ${CSV_PATH} \
  --model ${MODEL_PATH} \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE}

echo "All pruning runs completed!"
echo "Pruned models are available in the ${OUTPUT_DIR} directory."
