#!/bin/bash

# Script to compare layer-by-layer pruning with random pruning
# Creates comparison results for both approaches at 10% pruning
# Focused on layers 10-19 with 300 images

echo "Starting layer-by-layer pruning comparisons..."

# Create output directories if they don't exist
mkdir -p pruned_outputs/layer_pruning
mkdir -p comparison_results/layer_pruning

# Function to run comparison and check for errors
run_comparison() {
    local pruning_type=$1
    local layer_id=$2
    local percentage=$3
    local model_path=$4
    
    echo "========================================================"
    echo "Comparing original model with ${pruning_type}_layer${layer_id}_${percentage}..."
    echo "========================================================"
    
    python src/analysis/classify_with_models.py \
        --original-model data/models/resnet18.pth \
        --pruned-model "${model_path}" \
        --images data/unused_images \
        --model-type resnet18 \
        --output-dir "comparison_results/layer_pruning/models_vs_${pruning_type}_layer${layer_id}_${percentage}"
    
    if [ $? -eq 0 ]; then
        echo "Comparison with ${pruning_type}_layer${layer_id}_${percentage} completed successfully."
    else
        echo "Error comparing with ${pruning_type}_layer${layer_id}_${percentage}."
    fi
    echo ""
}

# Function to prune a specific layer and create model
prune_layer() {
    local layer_id=$1
    local percentage=$2
    local output_dir="pruned_outputs/layer_pruning/layer${layer_id}_${percentage}"
    
    echo "Pruning layer ${layer_id} at ${percentage}% using similarity scores..."
    mkdir -p "${output_dir}"
    
    python src/analysis/prune_specific_layer.py \
        --model data/models/resnet18.pth \
        --stats data/filter_stats.csv \
        --layer "${layer_id}" \
        --percentage "${percentage}" \
        --output "${output_dir}" \
        --model-type resnet18
    
    if [ $? -eq 0 ]; then
        echo "Layer ${layer_id} pruned successfully at ${percentage}%."
        return 0
    else
        echo "Error pruning layer ${layer_id}."
        return 1
    fi
}

# Function to create random pruning model for comparison
create_random_pruning() {
    local layer_id=$1
    local percentage=$2
    local output_dir="pruned_outputs/layer_pruning/random_layer${layer_id}_${percentage}"
    
    echo "Creating random pruning model for layer ${layer_id} at ${percentage}%..."
    mkdir -p "${output_dir}"
    
    python src/analysis/random_prune_specific_layer.py \
        --model data/models/resnet18.pth \
        --stats data/filter_stats.csv \
        --layer "${layer_id}" \
        --percentage "${percentage}" \
        --output "${output_dir}" \
        --model-type resnet18
    
    if [ $? -eq 0 ]; then
        echo "Random pruning for layer ${layer_id} created successfully at ${percentage}%."
        return 0
    else
        echo "Error creating random pruning for layer ${layer_id}."
        return 1
    fi
}

# Define layers to prune and percentages
layers=( 7 8 9)
percentages=(2 3 5)

# Run pruning and comparisons for each layer and percentage
for layer in "${layers[@]}"; do
    for percentage in "${percentages[@]}"; do
        echo "Processing layer ${layer} at ${percentage}%..."
        
        # Prune layer using similarity scores
        prune_layer "${layer}" "${percentage}"
        if [ $? -eq 0 ]; then
            run_comparison "similarity" "${layer}" "${percentage}" "pruned_outputs/layer_pruning/layer${layer}_${percentage}/pruned_model.pth"
        fi
        
        # Create random pruning for comparison
        create_random_pruning "${layer}" "${percentage}"
        if [ $? -eq 0 ]; then
            run_comparison "random" "${layer}" "${percentage}" "pruned_outputs/layer_pruning/random_layer${layer}_${percentage}/random_pruned_model.pth"
        fi
    done
done

echo "All layer-by-layer comparisons completed!"
echo "Results are available in the comparison_results/layer_pruning directory."

# Create summary plots
echo "Generating summary plots..."
python src/analysis/analyze_layer_pruning_results.py

echo "Done!"
