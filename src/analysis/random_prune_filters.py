import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# Import functions from prune_filters.py
from src.analysis.prune_filters import (
    load_model, map_layer_filter_to_pytorch, create_pruned_model, 
    save_pruned_model, visualize_pruning
)

def identify_random_filters_to_prune(stats_df, prune_percentage=0.3, random_seed=42):
    """
    Identify filters to prune randomly.

    Args:
        stats_df: DataFrame containing filter statistics
        prune_percentage: Percentage of filters to prune (0-1)
        random_seed: Random seed for reproducibility

    Returns:
        list: List of (layer_id, filter_id) tuples to prune
    """
    if stats_df is None or len(stats_df) == 0:
        print("No filter statistics available for pruning")
        return []
        
    # Make sure we have the right column names
    required_columns = ["Layer", "Filter"]
    missing_columns = [col for col in required_columns if col not in stats_df.columns]
    
    if missing_columns:
        # Try alternative column names
        alt_columns = {
            "Layer": "layer_id", 
            "Filter": "filter_id"
        }
        
        for req_col, alt_col in alt_columns.items():
            if req_col in missing_columns and alt_col in stats_df.columns:
                stats_df = stats_df.rename(columns={alt_col: req_col})
                missing_columns.remove(req_col)
    
    if missing_columns:
        print(f"Error: Missing required columns in stats file: {missing_columns}")
        return []

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Calculate number of filters to prune
    num_to_prune = int(len(stats_df) * prune_percentage)
    
    # Randomly select filters to prune
    random_indices = np.random.choice(len(stats_df), size=num_to_prune, replace=False)
    filters_to_prune = [(int(stats_df.iloc[i]['Layer']), int(stats_df.iloc[i]['Filter'])) 
                        for i in random_indices]
    
    print(f"Randomly identified {len(filters_to_prune)} filters to prune ({prune_percentage*100:.1f}% of total)")
    print(f"Using random seed: {random_seed}")

    return filters_to_prune

def main():
    parser = argparse.ArgumentParser(description='Randomly prune filters')
    parser.add_argument('--stats', type=str, required=True,
                        help='Path to the filter statistics CSV file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the original model file')
    parser.add_argument('--output', type=str, default='random_pruned_model',
                        help='Output directory for the pruned model')
    parser.add_argument('--percentage', type=float, default=0.3,
                        help='Percentage of filters to prune (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--model-type', type=str, default='resnet18',
                        help='Model architecture type')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load filter statistics directly from file
    print(f"Loading filter statistics from {args.stats}")
    try:
        stats_df = pd.read_csv(args.stats)
        print(f"Loaded statistics for {len(stats_df)} filters")
    except Exception as e:
        print(f"Error loading filter statistics: {e}")
        return
    
    if stats_df is None:
        print("Error: Could not load filter statistics. Exiting.")
        return

    # Identify filters to prune randomly
    filters_to_prune = identify_random_filters_to_prune(
        stats_df,
        prune_percentage=args.percentage,
        random_seed=args.seed
    )
    
    if not filters_to_prune:
        print("No filters identified for pruning. Exiting.")
        return

    # Load the original model
    original_model = load_model(args.model)
    
    if original_model is None:
        print("Error: Could not load original model. Exiting.")
        return

    # Create pruned model
    print("Creating randomly pruned model...")
    pruned_model = create_pruned_model(original_model, filters_to_prune, args.model_type)
    
    if pruned_model is None:
        print("Error: Could not create pruned model. Exiting.")
        return
    
    # Save the pruned model
    pruned_model_path = os.path.join(args.output, 'random_pruned_model.pth')
    save_pruned_model(pruned_model, pruned_model_path)

    # Add a "Random" column to the stats_df for visualization
    pruned_set = set(filters_to_prune)
    stats_df['Random'] = stats_df.apply(
        lambda row: (int(row['Layer']), int(row['Filter'])) in pruned_set,
        axis=1
    )
    
    # If there's an "Avg Similarity Score" column, use it for visualization
    if "Avg Similarity Score" not in stats_df.columns and "avg_similarity_score" in stats_df.columns:
        stats_df = stats_df.rename(columns={"avg_similarity_score": "Avg Similarity Score"})
    
    # If there's still no score column, add a dummy one for visualization
    if "Avg Similarity Score" not in stats_df.columns:
        stats_df["Avg Similarity Score"] = np.random.random(len(stats_df))
    
    # Visualize pruning
    visualize_pruning(
        stats_df,
        filters_to_prune,
        os.path.join(args.output, 'random_pruning_visualization.png')
    )

    # Save the list of pruned filters
    with open(os.path.join(args.output, 'random_pruned_filters.txt'), 'w') as f:
        f.write("Layer,Filter,Random\n")
        for layer_id, filter_id in filters_to_prune:
            f.write(f"{layer_id},{filter_id},1\n")

    print(f"\nRandomly pruned {len(filters_to_prune)} filters out of {len(stats_df)} total filters")
    print(f"Random seed used: {args.seed}")
    print(f"Pruning details saved to {os.path.join(args.output, 'random_pruned_filters.txt')}")
    print(f"Pruned model saved to {pruned_model_path}")
    print(f"\nUse classify_with_models.py to compare this random pruning with the original model")

if __name__ == "__main__":
    main()