#!/bin/bash

# Script to compare original model with all pruned models
# Creates comparison results for both random and similarity-based pruning at different percentages

echo "Starting model comparisons..."

# Create output directory if it doesn't exist
mkdir -p comparison_results

# Function to run comparison and check for errors
run_comparison() {
    local pruning_type=$1
    local percentage=$2
    local model_path=$3
    
    echo "========================================================"
    echo "Comparing original model with ${pruning_type}_${percentage}..."
    echo "========================================================"
    
    python src/analysis/classify_with_models.py \
        --original-model data/models/resnet18.pth \
        --pruned-model "${model_path}" \
        --images data/unused_images \
        --model-type resnet18 \
        --output-dir "comparison_results/models_vs_${pruning_type}_${percentage}"
    
    if [ $? -eq 0 ]; then
        echo "Comparison with ${pruning_type}_${percentage} completed successfully."
    else
        echo "Error comparing with ${pruning_type}_${percentage}."
    fi
    echo ""
}

# Random pruning comparisons
echo "Running random pruning comparisons..."
run_comparison "random" "1" "pruned_outputs/random_1/random_pruned_model.pth"
run_comparison "random" "2" "pruned_outputs/random_2/random_pruned_model.pth"
run_comparison "random" "3" "pruned_outputs/random_3/random_pruned_model.pth"
run_comparison "random" "10" "pruned_outputs/random_10/random_pruned_model.pth"
run_comparison "random" "20" "pruned_outputs/random_20/random_pruned_model.pth"
run_comparison "random" "30" "pruned_outputs/random_30/random_pruned_model.pth"

# Similarity-based pruning comparisons
echo "Running similarity-based pruning comparisons..."
run_comparison "similarity" "1" "pruned_outputs/similarity_1/pruned_model.pth"
run_comparison "similarity" "2" "pruned_outputs/similarity_2/pruned_model.pth"
run_comparison "similarity" "3" "pruned_outputs/similarity_3/pruned_model.pth"
run_comparison "similarity" "10" "pruned_outputs/similarity_10/pruned_model.pth"
run_comparison "similarity" "20" "pruned_outputs/similarity_20/pruned_model.pth"
run_comparison "similarity" "30" "pruned_outputs/similarity_30/pruned_model.pth"

echo "All comparisons completed!"
echo "Results are available in the comparison_results directory."