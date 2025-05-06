#!/bin/bash

# Script to compare all pruned models with the original model
# Uses images from the unused_images folder for comparison

echo "Starting comparison of all pruned models with original model..."

# Paths
ORIGINAL_MODEL="data/models/resnet18.pth"
PRUNED_MODELS_DIR="/Users/tansylu/Documents/ssvep/pruned_outputs"
IMAGES_DIR="data/unused_images"
OUTPUT_DIR="comparison_results/all_pruned_models"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run comparison
run_comparison() {
    local model_path=$1
    local model_name=$2
    
    echo "========================================================"
    echo "Comparing original model with ${model_name}..."
    echo "========================================================"
    
    python src/analysis/classify_with_models.py \
        --original-model "$ORIGINAL_MODEL" \
        --pruned-model "$model_path" \
        --images "$IMAGES_DIR" \
        --model-type resnet18 \
        --output-dir "$OUTPUT_DIR/${model_name}"
    
    if [ $? -eq 0 ]; then
        echo "Comparison with ${model_name} completed successfully."
    else
        echo "Error comparing with ${model_name}."
    fi
    echo ""
}

# Find all pruned models recursively
find "$PRUNED_MODELS_DIR" -name "pruned_model.pth" | while read -r model_path; do
    # Extract relative path for naming
    rel_path=$(echo "$model_path" | sed "s|$PRUNED_MODELS_DIR/||")
    dir_name=$(dirname "$rel_path")
    
    # Use directory name as model name
    model_name=$(basename "$(dirname "$model_path")")
    
    # Run comparison
    run_comparison "$model_path" "$model_name"
done

echo "All comparisons completed!"
echo "Results are available in the $OUTPUT_DIR directory."

# Generate summary of all comparisons
echo "Generating summary of all comparisons..."
python src/analysis/summarize_model_comparisons.py --results-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/summary.csv"