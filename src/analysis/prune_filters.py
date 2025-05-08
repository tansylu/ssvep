import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import json
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
from torchvision import models

'''
run with:
python src/analysis/prune_filters.py --model data/models/resnet18.pth --stats data/filter_stats.csv --model-type resnet18 --output pruned_output

For structural pruning (physically removing filters):
python src/analysis/prune_filters.py --model data/models/resnet18.pth --stats data/filter_stats.csv --model-type resnet18 --output structurally_pruned --percentage 0.05 --structural-pruning

For pruning and then retraining:
python src/analysis/prune_filters.py --model data/models/resnet18.pth --stats data/filter_stats.csv --model-type resnet18 --output pruned_retrained --percentage 0.05 --retrain --data-dir /path/to/dataset
'''

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

def load_filter_stats(stats_file):
    """Load filter statistics from CSV file"""
    print(f"Loading filter statistics from {stats_file}")
    try:
        stats_df = pd.read_csv(stats_file)
        print(f"Loaded statistics for {len(stats_df)} filters")
        return stats_df
    except Exception as e:
        print(f"Error loading filter statistics: {e}")
        return None

def identify_filters_to_prune(stats_df, prune_percentage=0.3, min_score=None, max_score=None, prune_highest=False):
    """
    Identify filters to prune based on similarity scores.

    Args:
        stats_df: DataFrame containing filter statistics
        prune_percentage: Percentage of filters to prune (0-1)
        min_score: Minimum similarity score threshold (filters below this will be pruned)
        max_score: Maximum similarity score threshold (filters above this will be pruned)
        prune_highest: If True, prune filters with highest scores instead of lowest

    Returns:
        list: List of (layer_id, filter_id) tuples to prune
    """
    if stats_df is None or len(stats_df) == 0:
        print("No filter statistics available for pruning")
        return []
        
    # Make sure we have the right column names
    required_columns = ["Layer", "Filter", "Avg Similarity Score"]
    missing_columns = [col for col in required_columns if col not in stats_df.columns]
    
    if missing_columns:
        # Try alternative column names
        alt_columns = {
            "Layer": "layer_id", 
            "Filter": "filter_id", 
            "Avg Similarity Score": "avg_similarity_score"
        }
        
        for req_col, alt_col in alt_columns.items():
            if req_col in missing_columns and alt_col in stats_df.columns:
                stats_df = stats_df.rename(columns={alt_col: req_col})
                missing_columns.remove(req_col)
    
    if missing_columns:
        print(f"Error: Missing required columns in stats file: {missing_columns}")
        return []

    # Sort by similarity score (ascending for lowest, descending for highest)
    sorted_df = stats_df.sort_values('Avg Similarity Score', ascending=not prune_highest)

    filters_to_prune = []

    # Prune based on score threshold
    if prune_highest and max_score is not None:
        filters_to_prune = [(int(row['Layer']), int(row['Filter']))
                           for _, row in stats_df.iterrows()
                           if row['Avg Similarity Score'] > max_score]
        print(f"Identified {len(filters_to_prune)} filters with score above {max_score}")
    elif not prune_highest and min_score is not None:
        filters_to_prune = [(int(row['Layer']), int(row['Filter']))
                           for _, row in stats_df.iterrows()
                           if row['Avg Similarity Score'] < min_score]
        print(f"Identified {len(filters_to_prune)} filters with score below {min_score}")
    # Prune based on percentage
    else:
        num_to_prune = int(len(sorted_df) * prune_percentage)
        filters_to_prune = [(int(row['Layer']), int(row['Filter']))
                           for _, row in sorted_df.head(num_to_prune).iterrows()]
        
        if prune_highest:
            print(f"Identified {len(filters_to_prune)} highest-scoring filters to prune ({prune_percentage*100:.1f}% of total)")
        else:
            print(f"Identified {len(filters_to_prune)} lowest-scoring filters to prune ({prune_percentage*100:.1f}% of total)")

    return filters_to_prune

def map_layer_filter_to_pytorch(layer_id, filter_id, model_type="resnet18"):
    """
    Map a database layer_id and filter_id to actual PyTorch model layer name.
    Add range checks to prevent out of range filter indices.
    """
    if model_type == "resnet18":
        # Layer sizes in ResNet18
        layer_sizes = {
            "conv1": 64,
            "layer1.0.conv1": 64, "layer1.0.conv2": 64,
            "layer1.1.conv1": 64, "layer1.1.conv2": 64,
            "layer2.0.conv1": 128, "layer2.0.conv2": 128,
            "layer2.1.conv1": 128, "layer2.1.conv2": 128,
            "layer3.0.conv1": 256, "layer3.0.conv2": 256,
            "layer3.1.conv1": 256, "layer3.1.conv2": 256,
            "layer4.0.conv1": 512, "layer4.0.conv2": 512,
            "layer4.1.conv1": 512, "layer4.1.conv2": 512,
            "fc": 1000,
            "bn1": 64
        }
        
        layer_map = {
            0: "conv1",                # First conv layer (64 filters)
            1: "layer1.0.conv1",       # First block, first conv (64 filters)
            2: "layer1.0.conv2",       # First block, second conv (64 filters)
            3: "layer1.1.conv1",       # Second block in layer1, first conv
            4: "layer1.1.conv2",       # Second block in layer1, second conv
            5: "layer2.0.conv1",       # First block in layer2, first conv (128 filters)
            6: "layer2.0.conv2",       # First block in layer2, second conv (128 filters)
            # Note: Layer 7 is missing in your system map
            8: "layer2.1.conv1",       # Second block in layer2, first conv (128 filters)
            9: "layer2.1.conv2",       # Second block in layer2, second conv (128 filters)
            10: "layer3.0.conv1",      # First block in layer3, first conv (256 filters)
            11: "layer3.0.conv2",      # First block in layer3, second conv (256 filters)
            # Note: Layer 12 is missing in your system map
            13: "layer3.1.conv1",      # Second block in layer3, first conv (256 filters)
            14: "layer3.1.conv2",      # Second block in layer3, second conv (256 filters)
            15: "layer4.0.conv1",      # First block in layer4, first conv (512 filters)
            16: "layer4.0.conv2",      # First block in layer4, second conv (512 filters)
            # Note: Layer 17 is missing in your system map
            18: "layer4.1.conv1",      # Second block in layer4, first conv (512 filters)
            19: "layer4.1.conv2"       # Second block in layer4, second conv (512 filters)
        }
        
        if layer_id in layer_map:
            module_name = layer_map[layer_id]
            
            # Check if filter_id is within the valid range for this layer
            if module_name in layer_sizes and filter_id < layer_sizes[module_name]:
                return module_name, filter_id
            else:
                print(f"Warning: Filter index {filter_id} out of range for {module_name} (max: {layer_sizes.get(module_name, 'unknown')})")
                return None, None
        else:
            print(f"Warning: Unknown layer_id {layer_id}")
            return None, None
    else:
        print(f"Warning: Unsupported model type {model_type}")
        return None, None

def load_model(model_path):
    """
    Load a PyTorch model from file.
    
    Args:
        model_path: Path to the PyTorch model file
        
    Returns:
        PyTorch model
    """
    try:
        device = torch.device("cpu")
        
        # Load the file
        loaded_obj = torch.load(model_path, map_location=device)
        
        # Check if it's a state dict
        if isinstance(loaded_obj, dict) and not isinstance(loaded_obj, torch.nn.Module):
            print(f"Loaded state dictionary from {model_path}, converting to model...")
            
            # Try to determine the model type
            if any("layer4.1.conv2" in k for k in loaded_obj.keys()):
                # Create a new model and load the state dict
                model = models.resnet18(weights=None)
                model.load_state_dict(loaded_obj, strict=False)
                print("Created ResNet18 model from state dictionary")
                return model
            elif any("layer4.2.conv3" in k for k in loaded_obj.keys()):
                model = models.resnet50(weights=None)
                model.load_state_dict(loaded_obj, strict=False)
                print("Created ResNet50 model from state dictionary")
                return model
            else:
                print("Could not determine model type from state dict")
                return None
        elif isinstance(loaded_obj, torch.nn.Module):
            print(f"Loaded full model from {model_path}")
            return loaded_obj
        else:
            print(f"Unexpected object type: {type(loaded_obj)}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_pruned_model(model, filters_to_prune, model_type="resnet18"):
    """Create a new model with pruned filters (weights zeroed out)."""
    try:
        # Check if model is a state dict
        is_state_dict = isinstance(model, dict) or hasattr(model, 'items')
        
        if is_state_dict:
            print("Model is a state dictionary, creating pruned state dict")
            # For state dict approach
            state_dict = model if isinstance(model, dict) else model.state_dict()
            
            # Create a new state dict with detached tensors
            pruned_state_dict = {}
            for key, tensor in state_dict.items():
                # Clone and detach each tensor to avoid gradient issues
                if isinstance(tensor, torch.Tensor):
                    pruned_state_dict[key] = tensor.clone().detach()
                else:
                    pruned_state_dict[key] = tensor
            
            pruned_filters_count = 0
            
            # Process each filter to prune
            for layer_id, filter_id in filters_to_prune:
                # Map database IDs to PyTorch layer name and filter index
                module_name, filter_index = map_layer_filter_to_pytorch(layer_id, filter_id, model_type)
                
                if module_name is None:
                    continue
                
                # For state dict, look for weights with this layer name
                weight_key = f"{module_name}.weight"
                bias_key = f"{module_name}.bias"
                
                if weight_key in pruned_state_dict:
                    # Zero out filter weights if within range
                    if filter_index < pruned_state_dict[weight_key].shape[0]:
                        pruned_state_dict[weight_key][filter_index] = 0.0
                        pruned_filters_count += 1
                        print(f"Pruned filter {filter_index} in {weight_key}")
                        
                        # Zero out bias if it exists
                        if bias_key in pruned_state_dict:
                            pruned_state_dict[bias_key][filter_index] = 0.0
                    else:
                        print(f"Warning: Filter index {filter_index} out of range for {weight_key}")
                else:
                    print(f"Warning: Could not find {weight_key} in state dict")
            
            print(f"Total filters pruned in state dict: {pruned_filters_count}")

            # Create a new model with the pruned weights
            if model_type == "resnet18":
                new_model = models.resnet18(weights=None)
                new_model.load_state_dict(pruned_state_dict, strict=False)
                return new_model
            elif model_type == "resnet50":
                new_model = models.resnet50(weights=None)
                new_model.load_state_dict(pruned_state_dict, strict=False)
                return new_model
            else:
                # Just return the state dict if we can't create a model
                print("Returning pruned state dictionary (no model created)")
                return pruned_state_dict
                
        else:
            # For full model approach
            print("Model is a full PyTorch model, creating pruned model")
            pruned_model = copy.deepcopy(model)
            
            # Dictionary to track which filters have been pruned in each module
            pruned_filters_by_module = {}
            
            # Process each filter to prune
            for layer_id, filter_id in filters_to_prune:
                # Map database IDs to PyTorch module and filter index
                module_name, filter_index = map_layer_filter_to_pytorch(layer_id, filter_id, model_type)
                
                if module_name is None:
                    continue
                    
                # Keep track of filters pruned in this module
                if module_name not in pruned_filters_by_module:
                    pruned_filters_by_module[module_name] = []
                pruned_filters_by_module[module_name].append(filter_index)
                
                print(f"Marking filter {filter_index} in module {module_name} for pruning")
                
            # Actually perform the pruning
            for module_name, filter_indices in pruned_filters_by_module.items():
                try:
                    # Get the module by name
                    module = pruned_model
                    for part in module_name.split('.'):
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                    
                    # Zero out the filters
                    for filter_idx in filter_indices:
                        if hasattr(module, 'weight') and isinstance(module, nn.Conv2d):
                            if filter_idx < module.weight.shape[0]:
                                with torch.no_grad():
                                    module.weight.data[filter_idx, :, :, :] = 0.0
                                    if module.bias is not None and filter_idx < module.bias.shape[0]:
                                        module.bias.data[filter_idx] = 0.0
                                        
                            print(f"Pruned filter {filter_idx} in {module_name}")
                        else:
                            print(f"Warning: Module {module_name} is not a Conv2d or has no weight attribute")
                except Exception as e:
                    print(f"Error pruning {module_name}: {e}")
                    continue

            print(f"Total filters pruned in model: {sum(len(filters) for filters in pruned_filters_by_module.values())}")
            return pruned_model
        
    except Exception as e:
        print(f"Error creating pruned model: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_compact_resnet(model, filters_to_prune, model_type="resnet18"):
    """
    Create a compact model by physically removing the pruned filters.
    
    Args:
        model: Original PyTorch model
        filters_to_prune: List of (layer_id, filter_id) tuples to prune
        model_type: Model architecture type
        
    Returns:
        Compact model with filters physically removed
    """
    # Group filters to prune by layer
    pruned_filters_by_module = {}
    for layer_id, filter_id in filters_to_prune:
        module_name, filter_index = map_layer_filter_to_pytorch(layer_id, filter_id, model_type)
        if module_name is None:
            continue
        
        if module_name not in pruned_filters_by_module:
            pruned_filters_by_module[module_name] = []
        
        pruned_filters_by_module[module_name].append(filter_index)
    
    # Sort filter indices for each module (important for correct removal)
    for module_name in pruned_filters_by_module:
        pruned_filters_by_module[module_name] = sorted(pruned_filters_by_module[module_name], reverse=True)
    
    print("Creating compact ResNet model with physically removed filters...")
    
    if model_type == "resnet18":
        # Create new ResNet model
        compact_model = models.resnet18(weights=None)
        
        # Copy weights for non-pruned filters
        with torch.no_grad():
            # First handle the initial conv layer
            if "conv1" in pruned_filters_by_module:
                pruned_filters = pruned_filters_by_module["conv1"]
                original_conv = model.conv1
                
                # Calculate the new number of filters
                new_out_channels = original_conv.out_channels - len(pruned_filters)
                
                # Create a new convolution layer with fewer filters
                new_conv = nn.Conv2d(
                    original_conv.in_channels, 
                    new_out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=(original_conv.bias is not None)
                )
                
                # Copy weights for the filters we're keeping
                keep_indices = [i for i in range(original_conv.out_channels) if i not in pruned_filters]
                for i, keep_idx in enumerate(keep_indices):
                    new_conv.weight.data[i] = original_conv.weight.data[keep_idx]
                    if original_conv.bias is not None:
                        new_conv.bias.data[i] = original_conv.bias.data[keep_idx]
                
                # Replace the layer in the model
                compact_model.conv1 = new_conv
                
                # Also adjust the batch normalization layer
                if hasattr(model, 'bn1') and hasattr(compact_model, 'bn1'):
                    orig_bn = model.bn1
                    new_bn = nn.BatchNorm2d(new_out_channels)
                    
                    # Copy batch norm parameters for kept filters
                    for i, idx in enumerate(keep_indices):
                        new_bn.weight.data[i] = orig_bn.weight.data[idx]
                        new_bn.bias.data[i] = orig_bn.bias.data[idx]
                        new_bn.running_mean.data[i] = orig_bn.running_mean.data[idx]
                        new_bn.running_var.data[i] = orig_bn.running_var.data[idx]
                    
                    compact_model.bn1 = new_bn
                
                print(f"Adjusted conv1: {original_conv.out_channels} -> {new_out_channels} filters")
            
            # Process each layer in the ResNet model
            for layer_idx in range(1, 5):  # layers 1 to 4 in ResNet
                layer_name = f"layer{layer_idx}"
                orig_layer = getattr(model, layer_name)
                compact_layer = getattr(compact_model, layer_name)
                
                for block_idx in range(len(orig_layer)):
                    orig_block = orig_layer[block_idx]
                    compact_block = compact_layer[block_idx]
                    
                    # Process first conv in the block
                    first_conv_name = f"{layer_name}.{block_idx}.conv1"
                    if first_conv_name in pruned_filters_by_module:
                        pruned_filters = pruned_filters_by_module[first_conv_name]
                        orig_conv = orig_block.conv1
                        
                        # Calculate new dimensions
                        new_out_channels = orig_conv.out_channels - len(pruned_filters)
                        
                        # Create new conv layer
                        new_conv = nn.Conv2d(
                            orig_conv.in_channels,
                            new_out_channels,
                            kernel_size=orig_conv.kernel_size,
                            stride=orig_conv.stride,
                            padding=orig_conv.padding,
                            bias=(orig_conv.bias is not None)
                        )
                        
                        # Copy weights for kept filters
                        keep_indices = [i for i in range(orig_conv.out_channels) if i not in pruned_filters]
                        for i, keep_idx in enumerate(keep_indices):
                            new_conv.weight.data[i] = orig_conv.weight.data[keep_idx]
                            if orig_conv.bias is not None:
                                new_conv.bias.data[i] = orig_conv.bias.data[keep_idx]
                        
                        # Replace the conv layer
                        compact_block.conv1 = new_conv
                        
                        # Adjust batch norm
                        orig_bn = orig_block.bn1
                        new_bn = nn.BatchNorm2d(new_out_channels)
                        
                        for i, idx in enumerate(keep_indices):
                            new_bn.weight.data[i] = orig_bn.weight.data[idx]
                            new_bn.bias.data[i] = orig_bn.bias.data[idx]
                            new_bn.running_mean.data[i] = orig_bn.running_mean.data[idx]
                            new_bn.running_var.data[i] = orig_bn.running_var.data[idx]
                        
                        compact_block.bn1 = new_bn
                        
                        print(f"Adjusted {first_conv_name}: {orig_conv.out_channels} -> {new_out_channels} filters")
                    
                    # Process second conv in the block
                    second_conv_name = f"{layer_name}.{block_idx}.conv2"
                    if second_conv_name in pruned_filters_by_module:
                        pruned_filters = pruned_filters_by_module[second_conv_name]
                        orig_conv = orig_block.conv2
                        
                        # Calculate new dimensions
                        new_out_channels = orig_conv.out_channels - len(pruned_filters)
                        new_in_channels = compact_block.conv1.out_channels  # Use the adjusted first conv's out_channels
                        
                        # Create new conv layer
                        new_conv = nn.Conv2d(
                            new_in_channels,
                            new_out_channels,
                            kernel_size=orig_conv.kernel_size,
                            stride=orig_conv.stride,
                            padding=orig_conv.padding,
                            bias=(orig_conv.bias is not None)
                        )
                        
                        # Copy weights for kept filters and adjusted input channels
                        keep_indices = [i for i in range(orig_conv.out_channels) if i not in pruned_filters]
                        prev_keep_indices = [i for i in range(orig_block.conv1.out_channels) 
                                          if i not in (pruned_filters_by_module.get(f"{layer_name}.{block_idx}.conv1") or [])]
                        
                        # Copy adjusted weights
                        for i, out_idx in enumerate(keep_indices):
                            for j, in_idx in enumerate(prev_keep_indices):
                                if j < new_in_channels and in_idx < orig_conv.weight.data.shape[1]:
                                    new_conv.weight.data[i, j] = orig_conv.weight.data[out_idx, in_idx]
                            
                            if orig_conv.bias is not None:
                                new_conv.bias.data[i] = orig_conv.bias.data[out_idx]
                        
                        # Replace the conv layer
                        compact_block.conv2 = new_conv
                        
                        # Adjust batch norm
                        orig_bn = orig_block.bn2
                        new_bn = nn.BatchNorm2d(new_out_channels)
                        
                        for i, idx in enumerate(keep_indices):
                            new_bn.weight.data[i] = orig_bn.weight.data[idx]
                            new_bn.bias.data[i] = orig_bn.bias.data[idx]
                            new_bn.running_mean.data[i] = orig_bn.running_mean.data[idx]
                            new_bn.running_var.data[i] = orig_bn.running_var.data[idx]
                        
                        compact_block.bn2 = new_bn
                        
                        print(f"Adjusted {second_conv_name}: {orig_conv.out_channels} -> {new_out_channels} filters")
                        
                    # Handle the downsample layer if present
                    if hasattr(orig_block, 'downsample') and orig_block.downsample is not None:
                        orig_downsample = orig_block.downsample[0]  # Downsample's conv
                        in_pruned = pruned_filters_by_module.get(f"{layer_name}.{block_idx-1}.conv2" if block_idx > 0 else "conv1", [])
                        out_pruned = pruned_filters_by_module.get(f"{layer_name}.{block_idx}.conv2", [])
                        
                        if in_pruned or out_pruned:
                            # Calculate new dimensions for downsample
                            new_in_channels = orig_downsample.in_channels - len(in_pruned) if in_pruned else orig_downsample.in_channels
                            new_out_channels = orig_downsample.out_channels - len(out_pruned) if out_pruned else orig_downsample.out_channels
                            
                            # Create new downsample conv
                            new_downsample = nn.Conv2d(
                                new_in_channels,
                                new_out_channels,
                                kernel_size=orig_downsample.kernel_size,
                                stride=orig_downsample.stride,
                                padding=orig_downsample.padding,
                                bias=(orig_downsample.bias is not None)
                            )
                            
                            # Copy weights for kept filters
                            in_keep_indices = [i for i in range(orig_downsample.in_channels) if i not in in_pruned]
                            out_keep_indices = [i for i in range(orig_downsample.out_channels) if i not in out_pruned]
                            
                            for i, out_idx in enumerate(out_keep_indices):
                                for j, in_idx in enumerate(in_keep_indices):
                                    if j < len(in_keep_indices) and in_idx < orig_downsample.weight.data.shape[1]:
                                        new_downsample.weight.data[i, j] = orig_downsample.weight.data[out_idx, in_idx]
                                
                                if orig_downsample.bias is not None:
                                    new_downsample.bias.data[i] = orig_downsample.bias.data[out_idx]
                            
                            # Create new downsample sequence
                            new_downsample_seq = nn.Sequential(
                                new_downsample,
                                nn.BatchNorm2d(new_out_channels)
                            )
                            
                            # Copy batch norm parameters
                            orig_bn = orig_block.downsample[1]
                            new_bn = new_downsample_seq[1]
                            
                            for i, idx in enumerate(out_keep_indices):
                                new_bn.weight.data[i] = orig_bn.weight.data[idx]
                                new_bn.bias.data[i] = orig_bn.bias.data[idx]
                                new_bn.running_mean.data[i] = orig_bn.running_mean.data[idx]
                                new_bn.running_var.data[i] = orig_bn.running_var.data[idx]
                                
                            # Replace the downsample layer
                            compact_block.downsample = new_downsample_seq
                            
                            print(f"Adjusted {layer_name}.{block_idx}.downsample: {orig_downsample.out_channels} -> {new_out_channels} filters")
            
            # Finally, adjust the fully-connected layer
            if hasattr(model, 'fc'):
                last_layer_name = "layer4.1.conv2"  # Last conv layer in ResNet18
                if last_layer_name in pruned_filters_by_module:
                    # Get the number of input features for the FC layer
                    orig_fc = model.fc
                    last_conv = getattr(compact_model.layer4[1], "conv2")
                    
                    # Get the number of output features from the last conv layer
                    last_conv_features = last_conv.out_channels
                    
                    # Calculate the new number of input features for FC layer
                    # Each filter in the last conv contributes to multiple features in the FC input
                    feature_multiplier = orig_fc.in_features // model.layer4[1].conv2.out_channels
                    new_in_features = last_conv_features * feature_multiplier
                    
                    # Create a new FC layer
                    new_fc = nn.Linear(new_in_features, orig_fc.out_features)
                    
                    # Copy weights for the filters we're keeping
                    # This is approximate as the exact mapping is complex after avgpool
                    pruned_filters = pruned_filters_by_module[last_layer_name]
                    kept_filters = [i for i in range(model.layer4[1].conv2.out_channels) if i not in pruned_filters]
                    
                    # For each kept filter, copy its contribution to the FC weights
                    for i, kept_idx in enumerate(kept_filters):
                        start_orig = kept_idx * feature_multiplier
                        start_new = i * feature_multiplier
                        
                        for j in range(feature_multiplier):
                            if start_orig + j < orig_fc.weight.data.shape[1] and start_new + j < new_fc.weight.data.shape[1]:
                                new_fc.weight.data[:, start_new + j] = orig_fc.weight.data[:, start_orig + j]
                    
                    # Copy bias
                    new_fc.bias.data = orig_fc.bias.data.clone()
                    
                    # Replace the FC layer
                    compact_model.fc = new_fc
                    
                    print(f"Adjusted FC layer: {orig_fc.in_features} -> {new_in_features} input features")
        
        print(f"Created compact ResNet18 with pruned filters physically removed")
        
        # Print summary of model size changes
        orig_params = sum(p.numel() for p in model.parameters())
        compact_params = sum(p.numel() for p in compact_model.parameters())
        reduction = (1 - compact_params / orig_params) * 100
        
        print(f"Original model parameters: {orig_params:,}")
        print(f"Compact model parameters: {compact_params:,}")
        print(f"Reduction: {reduction:.2f}%")
        
        return compact_model
    else:
        print(f"Error: Unsupported model type {model_type} for compact model creation")
        return None

def save_pruned_model(model, output_path, is_structural=False, architecture_info=None):
    """Save pruned model to a file, with architecture info if it's a structural pruning."""
    try:
        if is_structural and architecture_info is not None:
            # Create a package with both model and info
            save_dict = {
                'model': model,
                'model_info': architecture_info
            }
            
            torch.save(save_dict, output_path)
            print(f"Pruned model with architecture info saved to {output_path}")
        else:
            # Regular save
            torch.save(model, output_path)
            print(f"Pruned model saved to {output_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()

def visualize_pruning(stats_df, filters_to_prune, output_path):
    """
    Create a visualization of which filters were pruned.
    
    Args:
        stats_df: DataFrame containing filter statistics
        filters_to_prune: List of (layer_id, filter_id) tuples that were pruned
        output_path: Path to save the visualization
    """
    if stats_df is None:
        print("No statistics available for visualization")
        return
        
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

    # Set y-ticks to layer numbers
    plt.yticks(range(num_layers), [f'Layer {layer}' for layer in unique_layers])

    # Add labels and title
    plt.xlabel('Similarity Score')
    plt.ylabel('Layer')
    plt.title('Filter Pruning Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Pruning visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prune filters based on similarity scores')
    parser.add_argument('--stats', type=str, required=True,
                        help='Path to the filter statistics CSV file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the original model file')
    parser.add_argument('--output', type=str, default='pruned_model',
                        help='Output directory for the pruned model')
    parser.add_argument('--percentage', type=float, default=0.05,
                        help='Percentage of filters to prune (0-1)')
    parser.add_argument('--min-score', type=float, default=None,
                        help='Minimum similarity score threshold (filters below this will be pruned)')
    parser.add_argument('--max-score', type=float, default=None,
                        help='Maximum similarity score threshold (filters above this will be pruned)')
    parser.add_argument('--model-type', type=str, default='resnet18',
                        help='Model architecture type')
    parser.add_argument('--prune-highest', action='store_true',
                        help='Prune filters with highest scores instead of lowest')
    parser.add_argument('--structural-pruning', action='store_true',
                        help='Perform structural pruning (physically remove filters) instead of zeroing out weights')
    # Options for retraining (these will be passed to the retrain script)
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain the model after pruning')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for retraining')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for retraining')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load filter statistics directly from file
    stats_df = load_filter_stats(args.stats)
    
    if stats_df is None:
        print("Error: Could not load filter statistics. Exiting.")
        return

    # Identify filters to prune
    filters_to_prune = identify_filters_to_prune(
        stats_df,
        prune_percentage=args.percentage,
        min_score=args.min_score,
        max_score=args.max_score,
        prune_highest=args.prune_highest
    )
    
    if not filters_to_prune:
        print("No filters identified for pruning. Exiting.")
        return

    # Load the original model
    original_model = load_model(args.model)
    
    if original_model is None:
        print("Error: Could not load original model. Exiting.")
        return

    pruned_model_path = os.path.join(args.output, 'pruned_model.pth')

    # Create pruned model based on chosen method
    print("Creating pruned model...")
    if args.structural_pruning:
        pruned_model = create_compact_resnet(original_model, filters_to_prune, args.model_type)
        pruning_method = "structural pruning (filters physically removed)"
        
        # Create architecture information
        model_info = {
            'pruning_method': 'structural',
            'filters_pruned': len(filters_to_prune),
            'original_params': sum(p.numel() for p in original_model.parameters()),
            'pruned_params': sum(p.numel() for p in pruned_model.parameters()),
        }
        
        # Add layer information
        layer_info = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_info.append(f"{name}: {module.in_channels} in, {module.out_channels} out")
        
        model_info['layer_structure'] = layer_info
        
        # Save as JSON (keep this for backward compatibility)
        with open(os.path.join(args.output, 'model_architecture.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save model with architecture info
        save_pruned_model(pruned_model, pruned_model_path, is_structural=True, architecture_info=model_info)
    else:
        pruned_model = create_pruned_model(original_model, filters_to_prune, args.model_type)
        pruning_method = "weight zeroing (filters zeroed out)"
        save_pruned_model(pruned_model, pruned_model_path)
    
    if pruned_model is None:
        print("Error: Could not create pruned model. Exiting.")
        return
    
    # Save the pruned model before retraining
    pruned_model_path = os.path.join(args.output, 'pruned_model.pth')
    save_pruned_model(pruned_model, pruned_model_path)
    print(f"Pruned model saved to {pruned_model_path}")
    
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

    pruning_type = "highest-scoring" if args.prune_highest else "lowest-scoring"
    print(f"\nPruned {len(filters_to_prune)} {pruning_type} filters out of {len(stats_df)} total filters")
    print(f"Pruning method: {pruning_method}")
    
    # Save model architecture information to help with evaluation
    if args.structural_pruning:
        model_info = {
            'pruning_method': 'structural',
            'filters_pruned': len(filters_to_prune),
            'original_params': sum(p.numel() for p in original_model.parameters()),
            'pruned_params': sum(p.numel() for p in pruned_model.parameters()),
        }
        
        # Save a summary of layer changes
        layer_info = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_info.append(f"{name}: {module.in_channels} in, {module.out_channels} out")
        
        model_info['layer_structure'] = layer_info
        
        # Save as JSON
        with open(os.path.join(args.output, 'model_architecture.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
    
    # If retraining is requested, call the retrain script
    if args.retrain:
        if args.data_dir is None:
            print("Error: Must provide --data-dir for retraining. Skipping retraining step.")
        else:
            print("\nRetraining requested. Launching retrain_model.py...")
            import subprocess
            
            # Construct the retrain command
            retrain_cmd = [
                "python", f"{current_dir}/retrain_model.py",
                "--model", pruned_model_path,
                "--data-dir", args.data_dir,
                "--output", args.output,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size)
            ]
            
            # Execute the retraining command
            print(f"Running: {' '.join(retrain_cmd)}")
            try:
                subprocess.run(retrain_cmd, check=True)
                print("Retraining completed successfully!")
            except subprocess.CalledProcessError:
                print("Error during retraining process.")
            
            print("\nEvaluation instructions:")
            print(f"  To evaluate pruned (non-retrained) model: python src/analysis/evaluate_models.py --original {args.model} --pruned {pruned_model_path} --data [test_data_folder]")
            print(f"  To evaluate retrained model: python src/analysis/evaluate_models.py --original {args.model} --pruned {os.path.join(args.output, 'retrained_model.pth')} --data [test_data_folder]")
    else:
        print("\nEvaluation instructions:")
        print(f"  To evaluate pruned model: python src/analysis/evaluate_models.py --original {args.model} --pruned {pruned_model_path} --data [test_data_folder]")
    
    print("\nDone!")

if __name__ == "__main__":
    main()