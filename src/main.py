import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from flicker_image import flicker_image_and_save_gif
from model import ActivationModel, get_activations, save_activations
from PIL import Image
import numpy as np

def save_frames(frames, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_image = Image.fromarray(frame.astype(np.uint8))  # Convert numpy array to PIL Image
        frame_path = os.path.join(frames_dir, f"frame_{i}.png")
        frame_image.save(frame_path)
    print(f"Frames saved in '{frames_dir}' directory.")

def perform_activations(model, frames, preprocess_seqn, output_dir):
    if not os.path.exists(output_dir):
        activations = get_activations(model=model, frames=frames, preprocessing_sequence=preprocess_seqn)
        save_activations(activations=activations, output_dir=output_dir)
        print(f"Activations saved in '{output_dir}' directory.")
    else:
        print(f"Activations directory '{output_dir}' already exists. Skipping activation extraction.")

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
resnet18.eval()
print("Model architecture:")
print(resnet18)

# Define preprocessing transformations
print("Creating preprocessing sequence...")
preprocess_seqn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Generate flicker image and save as GIF
gif_paths = {
    "HSV": "flicker_hsv.gif",
    "LAB": "flicker_lab.gif",
    "RGB": "flicker_rgb.gif"
}

for color_format, gif_path in gif_paths.items():
    if not os.path.exists(gif_path):
        print(f"Generating flicker image and saving as GIF ({color_format})...")
        frames = flicker_image_and_save_gif(output_gif= gif_path,image_path='durov.jpg', frequency=5, duration=2, fps=30, color_format=color_format)
        
        # Save frames as images
        frames_dir = f"frames_{color_format.lower()}"
        save_frames(frames, frames_dir)
        
        print(f"GIF saved as '{gif_path}'.")

        # Perform activations for each color format
        activation_model = ActivationModel(resnet18)
        activations_output_dir = f'activations_output_{color_format.lower()}'
        perform_activations(activation_model, frames, preprocess_seqn, activations_output_dir)
    else:
        print(f"GIF '{gif_path}' already exists. Skipping generation.")