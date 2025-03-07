import torch
import numpy as np
import os
from PIL import Image
from torch.nn import Module
import matplotlib.pyplot as plt


# Extract all convolutional layers
class ActivationModel(Module):
    def __init__(self, model):
        super(ActivationModel, self).__init__()
        self.features = list(model.children())[:-2]  # Extract all layers except final FC layers
        self.model = torch.nn.Sequential(*self.features)
    
    def forward(self, x):
        activations = []
        for layer in self.model:
            x = layer(x)
            activations.append(x)
        return activations

def get_activations(*, model, frames, preprocessing_sequence):
    '''
    Extracts the activations of all layers of a model for a sequence of frames.
    Args:
        model: A PyTorch model.
        frames: A list of frames.
        preprocessing_sequence: A sequence of preprocessing transformations for an image.
    Returns:
        A dictionary where keys are layer indices and values are lists of activations for each frame.
    '''
    activations = {}
    for i, frame in enumerate(frames):
        print(f'Processing frame {i+1}/{len(frames)}')
        img = Image.fromarray(frame)
        x = preprocessing_sequence(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            layer_activations = model(x)  # Forward pass through the model
            print(f'Frame {i+1}:')
            for layer_idx, activation in enumerate(layer_activations):
                print(f'  Layer {layer_idx+1}: shape {activation.shape}')
                if layer_idx not in activations:
                    activations[layer_idx] = []
                activations[layer_idx].append(activation.cpu().numpy())
    print('Finished processing all frames.')
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
                mean_activation = np.mean(frame_activation[0, filter_idx, :, :])
                activation_strength.append(mean_activation)
            
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