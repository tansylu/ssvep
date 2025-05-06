import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import torch

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# Import functions from prune_filters.py
from src.analysis.prune_filters import (
    load_model, map_layer_filter_to_pytorch, create_pruned_model, 
    save_pruned_model, visualize_pruning
)

def identify_filters_to_prune_in_layer(stats_df, layer_id, prune_percentage=0.3):
    """
    Identify filters to prune in a specific layer based on similarity scores.

    Args:
        stats_df: DataFrame containing filter statistics
        layer_id: ID of the layer to prune
        prune_percentage: Percentage of filters in the layer to prune (0-1)

    Returns:
        list: List of (layer_id, filter_id) tuples to prune
    """
    if stats_df is None or len(stats_df) == 0:
        print("No filter statistics available for pruning")
        return []

    # Filter the DataFrame to only include the specified layer
    layer_df = stats_df[stats_df['Layer'] == layer_id].copy()
    
    if len(layer_df) == 0:
        print(f"No filters found for layer {layer_id}")
        return []
    
    # Sort by similarity score (ascending)
    sorted_df = layer_df.sort_values(by='Avg Similarity Score', ascending=True)
    
    # Calculate number of filters to prune in this layer
    num_to_prune = int(len(sorted_df) * prune_percentage)
    
    if num_to_prune == 0:
        print(f"No filters to prune in layer {layer_id} at {prune_percentage*100}% pruning rate")
        return []
    
    # Get the filters to prune (lowest similarity scores)
    filters_to_prune = [(int(row['Layer']), int(row['Filter'])) 
                        for _, row in sorted_df.head(num_to_prune).iterrows()]
    
    print(f"Identified {len(filters_to_prune)} filters to prune in layer {layer_id} ({prune_percentage*100:.1f}% of layer filters)")
    
    return filters_to_prune

def main():
    parser = argparse.ArgumentParser(description='Prune filters in a specific layer based on similarity scores')
    parser.add_argument('--stats', type=str, required=True,
                        help='Path to the filter statistics CSV file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the original model file')
    parser.add_argument('--layer', type=int, required=True,
                        help='Layer ID to prune')
    parser.add_argument('--output', type=str, default='pruned_layer_model',
                        help='Output directory for the pruned model')
    parser.add_argument('--percentage', type=float, default=0.3,
                        help='Percentage of filters in the layer to prune (0-1)')
    parser.add_argument('--model-type', type=str, default='resnet18',
                        help='Model architecture type')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load filter statistics
    try:
        stats_df = pd.read_csv(args.stats)
        print(f"Loaded filter statistics from {args.stats}")
    except Exception as e:
        print(f"Error loading filter statistics: {e}")
        return
    
    # Identify filters to prune in the specified layer
    filters_to_prune = identify_filters_to_prune_in_layer(
        stats_df, 
        args.layer, 
        prune_percentage=float(args.percentage)/100 if float(args.percentage) > 1 else float(args.percentage)
    )
    
    if not filters_to_prune:
        print(f"No filters identified for pruning in layer {args.layer}. Exiting.")
        return

    # Load the original model
    original_model = load_model(args.model)
    
    if original_model is None:
        print("Error: Could not load original model. Exiting.")
        return

    # Create pruned model
    print("Creating pruned model...")
    pruned_model = create_pruned_model(original_model, filters_to_prune, args.model_type)
    
    if pruned_model is None:
        print("Error: Could not create pruned model. Exiting.")
        return
    
    # Save the pruned model
    pruned_model_path = os.path.join(args.output, 'pruned_model.pth')
    save_pruned_model(pruned_model, pruned_model_path)

    # Add a "Pruned" column to the stats_df for visualization
    pruned_set = set(filters_to_prune)
    stats_df['Pruned'] = stats_df.apply(
        lambda row: (int(row['Layer']), int(row['Filter'])) in pruned_set,
        axis=1
    )

    # Visualize pruning
    visualize_pruning(
        stats_df,
        filters_to_prune,
        os.path.join(args.output, 'pruning_visualization.png')
    )

    # Save the list of pruned filters
    with open(os.path.join(args.output, 'pruned_filters.txt'), 'w') as f:
        f.write("Layer,Filter,Score\n")
        for layer_id, filter_id in filters_to_prune:
            # Find the score for this filter
            filter_row = stats_df[(stats_df['Layer'] == layer_id) & (stats_df['Filter'] == filter_id)]
            if not filter_row.empty:
                score = filter_row['Avg Similarity Score'].values[0]
                f.write(f"{layer_id},{filter_id},{score:.6f}\n")
            else:
                f.write(f"{layer_id},{filter_id},unknown\n")

    print(f"\nPruned {len(filters_to_prune)} filters in layer {args.layer}")
    print(f"Pruning details saved to {os.path.join(args.output, 'pruned_filters.txt')}")
    print(f"Pruned model saved to {pruned_model_path}")

if __name__ == "__main__":
    main()