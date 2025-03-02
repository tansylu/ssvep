import torch
import numpy as np
import cv2
import os
from PIL import Image

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


def _get_activations(*, model, frames, preprocessing_sequence, color_space_func=cv2.COLOR_BGR2RGB):
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
    for frame in frames:
        img_array = cv2.cvtColor(frame, color_space_func)
        img = Image.fromarray(img_array)
        x = preprocessing_sequence(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            layer_activations = model(x)  # Forward pass through the model
        activations.append(layer_activations)
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
        for layer_idx, layer_activation in enumerate(frame_activations):
            np.save(os.path.join(output_dir, f'frame_{i}_layer_{layer_idx}.npy'), layer_activation.cpu().numpy())

