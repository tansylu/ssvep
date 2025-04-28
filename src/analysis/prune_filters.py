import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# Import necessary modules
from src.database import db

def load_filter_stats(stats_file):
    """Load filter statistics from CSV file"""
    print(f"Loading filter statistics from {stats_file}")
    stats_df = pd.read_csv(stats_file)
    return stats_df

def identify_filters_to_prune(stats_df, prune_percentage=0.3, min_score=None):
    """
    Identify filters to prune based on similarity scores.

    Args:
        stats_df: DataFrame containing filter statistics
        prune_percentage: Percentage of worst filters to prune (0-1)
        min_score: Minimum similarity score threshold (filters below this will be pruned)

    Returns:
        list: List of (layer_id, filter_id) tuples to prune
    """
    # Sort by similarity score (ascending)
    sorted_df = stats_df.sort_values('Avg Similarity Score', ascending=True)

    filters_to_prune = []

    # Prune based on minimum score threshold
    if min_score is not None:
        filters_to_prune = [(int(row['Layer']), int(row['Filter']))
                           for _, row in stats_df.iterrows()
                           if row['Avg Similarity Score'] < min_score]
        print(f"Identified {len(filters_to_prune)} filters with score below {min_score}")

    # Prune based on percentage
    else:
        num_to_prune = int(len(sorted_df) * prune_percentage)
        filters_to_prune = [(int(row['Layer']), int(row['Filter']))
                           for _, row in sorted_df.head(num_to_prune).iterrows()]
        print(f"Identified {len(filters_to_prune)} filters to prune ({prune_percentage*100:.1f}% of total)")

    return filters_to_prune

def load_model_weights(model_path):
    """
    Load model weights from a PyTorch .pth file.

    Args:
        model_path: Path to the PyTorch model file (.pth)

    Returns:
        dict: Dictionary of layer names and their weights
    """
    try:
        # Try to load using torch if available
        try:
            import torch
            model_weights = torch.load(model_path, map_location='cpu')
            print(f"Loaded PyTorch model weights from {model_path} using torch")
            return model_weights
        except ImportError:
            # If torch is not available, try to load using numpy
            print("PyTorch not available, trying to load using numpy...")
            model_weights = np.load(model_path, allow_pickle=True)
            print(f"Loaded model weights from {model_path} using numpy")
            return model_weights
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_pruned_weights(model_weights, filters_to_prune, layer_name_mapping=None):
    """
    Create pruned weights by setting specified filters to zero.

    Args:
        model_weights: Dictionary of layer names and their weights (PyTorch state_dict)
        filters_to_prune: List of (layer_id, filter_id) tuples to prune
        layer_name_mapping: Optional mapping from layer_id to layer name

    Returns:
        dict: Dictionary of pruned weights
    """
    # Try to import torch for tensor operations
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    # Create a deep copy of the weights
    if has_torch and isinstance(model_weights, dict) and any(isinstance(v, torch.Tensor) for v in model_weights.values()):
        # This is a PyTorch state_dict
        pruned_weights = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model_weights.items()}
        print("Processing PyTorch state_dict")
    else:
        # This is a regular dictionary of numpy arrays or other objects
        pruned_weights = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in model_weights.items()}
        print("Processing generic weight dictionary")

    # Group filters by layer
    filters_by_layer = {}
    for layer_id, filter_id in filters_to_prune:
        if layer_id not in filters_by_layer:
            filters_by_layer[layer_id] = []
        filters_by_layer[layer_id].append(filter_id)

    # Process each layer's weights
    pruned_filters_count = 0

    # Print the keys in the model weights to help with debugging
    print("Available keys in model weights:")
    for key in pruned_weights.keys():
        print(f"  {key}")

    # For PyTorch models, we need to look for layer names like 'layer1.0.conv1.weight'
    for layer_id, filter_ids in filters_by_layer.items():
        # Try different naming patterns for the layer
        layer_patterns = [
            f"layer{layer_id}",  # ResNet style
            f"conv{layer_id}",   # Simple CNN style
            f"features.{layer_id}",  # VGG style
            f"conv2d_{layer_id}"  # TensorFlow style
        ]

        if layer_name_mapping is not None and layer_id in layer_name_mapping:
            layer_patterns.insert(0, layer_name_mapping[layer_id])  # Use custom mapping if provided

        # Look for weights for this layer
        found_layer = False

        for pattern in layer_patterns:
            # Look for weight keys matching this pattern
            weight_keys = [k for k in pruned_weights.keys() if pattern in k and ('.weight' in k or 'kernel' in k)]

            for weights_key in weight_keys:
                # Find corresponding bias key if it exists
                bias_key = weights_key.replace('weight', 'bias').replace('kernel', 'bias')
                if bias_key not in pruned_weights:
                    bias_key = None

                # Get the weights
                weights = pruned_weights[weights_key]

                # Handle PyTorch tensors or numpy arrays
                if has_torch and isinstance(weights, torch.Tensor):
                    # For PyTorch tensors, the filter dimension is typically the first dimension for conv layers
                    if len(weights.shape) >= 2:  # Conv layers have at least 2D weights
                        for filter_id in filter_ids:
                            if filter_id < weights.shape[0]:  # Check if filter_id is valid
                                # Zero out the filter weights
                                weights[filter_id, ...] = 0
                                pruned_filters_count += 1

                                # Zero out the bias if it exists
                                if bias_key is not None and filter_id < pruned_weights[bias_key].shape[0]:
                                    pruned_weights[bias_key][filter_id] = 0

                        found_layer = True
                        print(f"Pruned {len(filter_ids)} filters from layer {layer_id} ({weights_key})")
                else:
                    # For numpy arrays, we need to determine the filter dimension
                    if len(weights.shape) >= 2:  # Conv layers have at least 2D weights
                        # Typically the last dimension for TensorFlow-style weights, first for PyTorch-style
                        filter_dim = 0 if weights.shape[0] > weights.shape[-1] else -1

                        for filter_id in filter_ids:
                            if filter_dim == -1 and filter_id < weights.shape[-1]:
                                # TensorFlow-style: filters are the last dimension
                                weights[..., filter_id] = 0
                                pruned_filters_count += 1

                                # Zero out the bias if it exists
                                if bias_key is not None and filter_id < pruned_weights[bias_key].shape[-1]:
                                    pruned_weights[bias_key][filter_id] = 0

                            elif filter_dim == 0 and filter_id < weights.shape[0]:
                                # PyTorch-style: filters are the first dimension
                                weights[filter_id, ...] = 0
                                pruned_filters_count += 1

                                # Zero out the bias if it exists
                                if bias_key is not None and filter_id < pruned_weights[bias_key].shape[0]:
                                    pruned_weights[bias_key][filter_id] = 0

                        found_layer = True
                        print(f"Pruned {len(filter_ids)} filters from layer {layer_id} ({weights_key})")

            if found_layer:
                break

        if not found_layer:
            print(f"Could not find weights for layer {layer_id}, skipping")

    print(f"Total pruned filters: {pruned_filters_count}")
    return pruned_weights

def save_pruned_weights(weights, output_path):
    """
    Save pruned weights to a file.

    Args:
        weights: Dictionary of pruned weights
        output_path: Path to save the weights
    """
    # Try to import torch for tensor operations
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    # Check if this is a PyTorch state_dict
    if has_torch and isinstance(weights, dict) and any(isinstance(v, torch.Tensor) for v in weights.values()):
        # Save as a PyTorch model
        torch.save(weights, output_path)
        print(f"Pruned PyTorch weights saved to {output_path}")
    else:
        # Save as a compressed NPZ file
        # Convert any non-numpy arrays to numpy arrays
        numpy_weights = {}
        for k, v in weights.items():
            if isinstance(v, np.ndarray):
                numpy_weights[k] = v
            else:
                try:
                    numpy_weights[k] = np.array(v)
                except:
                    print(f"Warning: Could not convert {k} to numpy array, skipping")

        np.savez_compressed(output_path, **numpy_weights)
        print(f"Pruned weights saved to {output_path}")

def visualize_pruning(stats_df, filters_to_prune, output_path):
    """
    Create a visualization of which filters were pruned using a dot plot.

    Args:
        stats_df: DataFrame containing filter statistics
        filters_to_prune: List of (layer_id, filter_id) tuples that were pruned
        output_path: Path to save the visualization
    """
    # Convert filters_to_prune to a set for faster lookup
    pruned_set = set(filters_to_prune)

    # Add a 'Pruned' column to the DataFrame
    stats_df['Pruned'] = stats_df.apply(
        lambda row: (int(row['Layer']), int(row['Filter'])) in pruned_set,
        axis=1
    )

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Get unique layers
    unique_layers = sorted(stats_df['Layer'].unique())
    num_layers = len(unique_layers)

    # Create a mapping from layer to y-position
    layer_to_y = {layer: i for i, layer in enumerate(unique_layers)}

    # Plot pruned filters in red, kept filters in blue
    pruned_df = stats_df[stats_df['Pruned']]
    kept_df = stats_df[~stats_df['Pruned']]

    # Sort dataframes by similarity score for clearer visualization
    pruned_df = pruned_df.sort_values('Avg Similarity Score')
    kept_df = kept_df.sort_values('Avg Similarity Score')

    # Add jitter to y-positions to avoid overlapping points
    def add_jitter(layer_series):
        return layer_series.map(layer_to_y) + np.random.uniform(-0.3, 0.3, size=len(layer_series))

    # Plot kept filters
    plt.scatter(
        kept_df['Avg Similarity Score'],
        add_jitter(kept_df['Layer']),
        color='blue', alpha=0.6, s=20, label='Kept Filters'
    )

    # Plot pruned filters
    plt.scatter(
        pruned_df['Avg Similarity Score'],
        add_jitter(pruned_df['Layer']),
        color='red', alpha=0.6, s=20, label='Pruned Filters'
    )

    # Add a vertical line at the pruning threshold
    if not pruned_df.empty:
        threshold = pruned_df['Avg Similarity Score'].max()
        plt.axvline(x=threshold, color='red', linestyle='--',
                    label=f'Pruning Threshold: {threshold:.4f}')

        # Add text to explain the pruning method
        if len(pruned_df) == int(len(stats_df) * 0.3):  # If using percentage-based pruning
            plt.text(0.5, 0.95, f'Pruning Method: Bottom {len(pruned_df)} filters (30%)',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        else:  # If using threshold-based pruning
            plt.text(0.5, 0.95, f'Pruning Method: All filters below {threshold:.4f}',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Set y-ticks to layer numbers
    plt.yticks(range(num_layers), [f'Layer {layer}' for layer in unique_layers])

    # Add labels and title
    plt.xlabel('Similarity Score')
    plt.ylabel('Layer')
    plt.title('Filter Pruning Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add a histogram of similarity scores at the bottom
    plt.figure(figsize=(10, 4))

    # Create bins for the histogram
    min_score = stats_df['Avg Similarity Score'].min()
    max_score = stats_df['Avg Similarity Score'].max()
    bins = np.linspace(min_score, max_score, 100)

    # Plot histogram of kept scores in blue
    plt.hist(kept_df['Avg Similarity Score'], bins=bins, alpha=0.5, color='blue', label='Kept Filters')

    # Plot histogram of pruned scores in red
    if not pruned_df.empty:
        plt.hist(pruned_df['Avg Similarity Score'], bins=bins, alpha=0.5, color='red', label='Pruned Filters')
        threshold = pruned_df['Avg Similarity Score'].max()
        plt.axvline(x=threshold, color='red', linestyle='--',
                    label=f'Pruning Threshold: {threshold:.4f}')

        # Add text to explain the pruning method
        if len(pruned_df) == int(len(stats_df) * 0.3):  # If using percentage-based pruning
            plt.text(0.5, 0.9, f'Pruning Method: Bottom {len(pruned_df)} filters (30%)',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        else:  # If using threshold-based pruning
            plt.text(0.5, 0.9, f'Pruning Method: All filters below {threshold:.4f}',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figures
    plt.figure(1)
    plt.tight_layout()
    plt.savefig(output_path)

    plt.figure(2)
    plt.tight_layout()
    histogram_path = output_path.replace('.png', '_histogram.png')
    plt.savefig(histogram_path)

    print(f"Pruning visualization saved to {output_path}")
    print(f"Score histogram saved to {histogram_path}")

def main():
    parser = argparse.ArgumentParser(description='Prune filters based on similarity scores')
    parser.add_argument('--stats', type=str, default='filter_stats.csv',
                        help='Path to the filter statistics CSV file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the original model file')
    parser.add_argument('--output', type=str, default='pruned_model',
                        help='Output directory for the pruned model')
    parser.add_argument('--percentage', type=float, default=0.3,
                        help='Percentage of worst filters to prune (0-1)')
    parser.add_argument('--min-score', type=float, default=None,
                        help='Minimum similarity score threshold (filters below this will be pruned)')
    parser.add_argument('--layer-mapping', type=str, default=None,
                        help='Path to a CSV file mapping layer IDs to layer names')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load filter statistics
    stats_df = load_filter_stats(args.stats)

    # Identify filters to prune
    filters_to_prune = identify_filters_to_prune(
        stats_df,
        prune_percentage=args.percentage,
        min_score=args.min_score
    )

    # Load layer mapping if provided
    layer_name_mapping = None
    if args.layer_mapping:
        try:
            mapping_df = pd.read_csv(args.layer_mapping)
            layer_name_mapping = dict(zip(mapping_df['layer_id'], mapping_df['layer_name']))
            print(f"Loaded layer mapping from {args.layer_mapping}")
        except Exception as e:
            print(f"Error loading layer mapping: {e}")

    # Load the original model weights
    model_weights = load_model_weights(args.model)

    if model_weights is None:
        print("Failed to load model weights. Exiting.")
        return

    # Create pruned weights
    pruned_weights = create_pruned_weights(model_weights, filters_to_prune, layer_name_mapping)

    # Save the pruned weights
    # Use .pth extension for PyTorch models
    save_pruned_weights(pruned_weights, os.path.join(args.output, 'pruned_weights.pth'))

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

    print(f"Pruned {len(filters_to_prune)} filters out of {len(stats_df)} total filters")
    print(f"Pruning details saved to {os.path.join(args.output, 'pruned_filters.txt')}")
    print(f"To use the pruned weights, you'll need to load them in your model code.")

if __name__ == "__main__":
    main()
