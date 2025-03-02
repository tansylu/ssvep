import torch
import numpy as np
import cv2
import os
from PIL import Image
from torch.nn import Module

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


def get_activations(*, model, frames, preprocessing_sequence,):
    '''
    Extracts the activations of all layers of a model for a sequence of frames.
    Args:
        model: A PyTorch model.
        frames: A list of frames.
        preprocessing_sequence: A sequence of preprocessing transformations for an image.
        color_space_func: A function that converts the color space of the frames.
    Returns:
        A list of activations for each layer of the model.
    '''
    activations = []
    for i, frame in enumerate(frames):
        print(f'Processing frame {i+1}/{len(frames)}')
        img = Image.fromarray(frame)
        x = preprocessing_sequence(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            layer_activations = model(x)  # Forward pass through the model
        activations.append(layer_activations)
    print('Finished processing all frames.')
    return activations

def save_activations(*, activations, output_dir):
    '''
    Saves the activations of all layers of a model for a sequence of frames.
    Args:
        activations: A list of activations for each layer of the model.
        output_dir: The directory where the activations will be saved.
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame_activations in enumerate(activations):
        print(f'Saving activations for frame {i+1}/{len(activations)}')
        for layer_idx, layer_activation in enumerate(frame_activations):
            np.save(os.path.join(output_dir, f'frame_{i}_layer_{layer_idx}.npy'), layer_activation.cpu().numpy())
    print('Finished saving all activations.')