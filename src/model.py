import torch
import numpy as np
import os
from PIL import Image
from torch.nn import Module
import matplotlib.pyplot as plt
import torchvision.models as models
import urllib.request

def init_model():# use once to set weigths and load the model.
    # Load ResNet18 model
    print("Loading ResNet18 model...")
    resnet18 = models.resnet18()

    # Define path to weights file
    weights_path = 'resnet18.pth'
    weights_only_path = 'resnet18_weights_only.pth'

    if not os.path.exists(weights_only_path):
        print(f"Loading model weights from {weights_path}...")

        # Try loading the model weights
        try:
            checkpoint = torch.load(weights_path, weights_only=False)  # Allow full loading for legacy formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Detected full checkpoint. Extracting model weights...")
                checkpoint = checkpoint['model_state_dict']
            resnet18.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            exit(1)

        # Save in a pure weights-only format for future compatibility
        torch.save(resnet18.state_dict(), weights_only_path)
        print(f"Converted and saved weights-only file: '{weights_only_path}'")
    else:
        print(f"Weights-only file '{weights_only_path}' already exists. Skipping loading and saving weights.")

    # Set model to evaluation mode
    print("Setting model to evaluation mode...")

    resnet18.eval()#sets the model into evalutaion mode which freezes BatchNorm & Dropout

    return resnet18




# Function to extract activations using hooks

# Global variable to track if filter counts have been printed
_filter_counts_printed = False

def get_activations(*, model, frames, preprocessing_sequence):
    '''
    Extracts the activations of all layers of a model for a sequence of frames using hooks.
    Args:
        model: A PyTorch model.
        frames: A list of frames.
        preprocessing_sequence: A sequence of preprocessing transformations for an image.
    Returns:
        A dictionary where keys are layer indices and values are lists of activations for each frame.
    '''
    # Dictionary to store activations
    activations = {}
    hooks = []
    layer_idx_map = {}

    # Define hook function
    def hook_fn(layer_idx):
        def _hook(_module, _input, output):
            # Store the output of the layer
            if layer_idx not in activations:
                activations[layer_idx] = []
            # Convert to numpy and store
            activations[layer_idx].append(output.detach().cpu().numpy())
        return _hook

    # Register hooks for layers we're interested in
    idx = 0
    for name, module in model.named_modules():
        # Only include Conv2d layers (exclude Linear/FC layers)
        if isinstance(module, torch.nn.Conv2d):
            # Skip the first convolutional layer (layer 0) and downsample layers (7, 12, 17)
            if  'downsample' not in name:
                layer_idx_map[name] = idx
                hooks.append(module.register_forward_hook(hook_fn(idx)))
            idx += 1

    # Process each frame
    for frame_idx, frame in enumerate(frames):
        img = Image.fromarray(frame)
        x = preprocessing_sequence(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # disable gradient computation
            _ = model(x)  # Forward pass through the model

        if frame_idx % 100 == 0 and frame_idx > 0:
            print(f"Processed {frame_idx}/{len(frames)} frames")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print layer mapping for reference with number of filters (only once)
    global _filter_counts_printed
    if not _filter_counts_printed:
        print("Layer index mapping:")

        # Create a dictionary to store filter counts for each layer
        layer_filter_counts = {}

        # Count filters in each layer's activation
        for layer_idx, layer_activations in activations.items():
            if layer_activations and len(layer_activations) > 0:
                # Get the number of filters from the first frame's activation
                # Shape is typically [batch_size, num_filters, height, width]
                num_filters = layer_activations[0].shape[1]
                layer_filter_counts[layer_idx] = num_filters

        # Print layer mapping with filter counts
        for name, idx in layer_idx_map.items():
            num_filters = layer_filter_counts.get(idx, 0)
            print(f"  Layer {idx}: {name} - {num_filters} filters")

        # Set the flag to True so we don't print again
        _filter_counts_printed = True

    return activations


def plot_activations(activations, output_dir):
    '''
    Plots and saves the activations of all layers of a model over frames.
    Args:
        activations: A dictionary where keys are layer indices and values are lists of activations for each frame.
        output_dir: The directory where the plots will be saved.
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for layer_idx, layer_activations in activations.items():
        for filter_idx in range(layer_activations[0].shape[1]):
            activation_strength = []
            for frame_activation in layer_activations:
                max_activation = np.max(frame_activation[0, filter_idx, :, :])#change to l2-norm.
                activation_strength.append(max_activation)

            plt.figure(figsize=(10, 5))
            plt.plot(activation_strength, label=f'Filter {filter_idx}')
            plt.title(f'Layer {layer_idx+1} Filter {filter_idx} Activations Over Frames')
            plt.xlabel('Frame #')
            plt.ylabel('Activation Strength')
            plt.legend()
            plot_path = os.path.join(output_dir, f'layer_{layer_idx}_filter_{filter_idx}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved plot for Layer {layer_idx+1} Filter {filter_idx} at {plot_path}')

def save_activations(*, activations, output_dir):
    '''
    Saves the activations of all layers of a model for a sequence of frames.
    Args:
        activations: A dictionary where keys are layer indices and values are lists of activations for each frame.
        output_dir: The directory where the activations will be saved.
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for layer_idx, layer_activations in activations.items():
        for frame_idx, frame_activation in enumerate(layer_activations):
            print(f'Saving activations for frame {frame_idx+1}/{len(layer_activations)}, layer {layer_idx+1}')
            np.save(os.path.join(output_dir, f'frame_{frame_idx}_layer_{layer_idx}.npy'), frame_activation)
    print('Finished saving all activations.')

def load_activations(output_dir):
    '''
    Loads the activations of all layers of a model for a sequence of frames from the specified directory.
    Args:
        output_dir: The directory where the activations are saved.
    Returns:
        A dictionary where keys are layer indices and values are lists of activations for each frame.
    '''
    activations = {}
    for file_name in sorted(os.listdir(output_dir)):
        if file_name.endswith('.npy'):
            parts = file_name.split('_')
            frame_idx = int(parts[1])
            layer_idx = int(parts[3].split('.')[0])
            activation = np.load(os.path.join(output_dir, file_name))
            if layer_idx not in activations:
                activations[layer_idx] = []
            while len(activations[layer_idx]) <= frame_idx:
                activations[layer_idx].append(None)
            activations[layer_idx][frame_idx] = activation
    return activations

def reduce_activation(activation, method='l2'):
    """
    Reduces a filter's activation map to a single value using specified method.
    Args:
        activation: numpy array of shape (H, W) - spatial dimensions of filter activation
        method: string, reduction method:
            - 'l1': L1 norm (sum of absolute values)
            - 'l2': L2 norm (sqrt of sum of squares)
            - 'mean': average of all values
            - 'max': maximum value
            - 'min': minimum value
            - 'std': standard deviation
            - 'median': median value
            - 'rms': root mean square (L2 normalized)
            - 'energy': sum of squared values (L2 norm squared)
    Returns:
        float: single value representing filter's activation
    """
    if method == 'l2':
        return np.sqrt(np.sum(activation ** 2))
    elif method == 'l1':
        return np.sum(np.abs(activation))
    elif method == 'mean':
        return np.mean(activation)
    elif method == 'max':
        return np.max(activation)
    elif method == 'min':
        return np.min(activation)
    elif method == 'std':
        return np.std(activation)
    elif method == 'median':
        return np.median(activation)
    elif method == 'rms':
        return np.sqrt(np.mean(activation ** 2))
    elif method == 'energy':
        return np.sum(activation ** 2)
    else:
        raise ValueError(f"Unknown reduction method: {method}. Valid methods are: "
                       "'l1', 'l2', 'mean', 'max', 'min', 'std', 'median', 'rms', 'energy'")
