import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from flicker_image import flicker_image_and_save_gif
from model import ActivationModel, get_activations, load_activations, save_activations, plot_activations
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import csv

def save_frames(frames, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_image = Image.fromarray(frame.astype(np.uint8))  # Convert numpy array to PIL Image
        frame_path = os.path.join(frames_dir, f"frame_{i}.png")
        frame_image.save(frame_path)
    print(f"Frames saved in '{frames_dir}' directory.")

def load_frames(frames_dir):
    frames = []
    for frame_file in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_file)
        frame_image = Image.open(frame_path)
        frames.append(np.array(frame_image))
    return frames

def perform_activations(model, frames, preprocess_seqn):
    activations = get_activations(model=model, frames=frames, preprocessing_sequence=preprocess_seqn)
    return activations

def perform_fourier_transform(activations):
    """
    Performs FFT on activation time series for each layer and filter.
    Args:
        activations: {layer_id: [frame1_tensor(1,filters,height,width), frame2_tensor...]}
    Returns:
        {layer_id: numpy_array(num_filters, fft_length)}
    """
    fourier_transformed_activations = {}
    for layer_id, frames in activations.items():
        num_filters = frames[0].shape[1]
        num_frames = len(frames)
        fourier_transformed_activations[layer_id] = np.zeros((num_filters, num_frames))
        for filter_id  in range(num_filters):
            # Get temporal sequence (the activation values of a single filter across frames)
            temporal_sequence = [np.mean(frame[0, filter_id, :, :]) for frame in frames]
            # Perform Fourier Transform on each filter's activations
            fourier_transformed_activations[layer_id][filter_id] = np.abs(fft(temporal_sequence))
    return fourier_transformed_activations

def find_dominant_frequencies(fourier_transformed_activations, fps=30):
    """
    Args:
        fourier_transformed_activations: {layer_id: np.array(num_filters, fft_length)}
        fps: sampling rate in Hz
    Returns:
        {layer_id: {filter_id: dominant_frequency}}
    """
    dominant_frequencies = {}
    for layer_id, layer_fft in fourier_transformed_activations.items():
        num_filters, fft_length = layer_fft.shape
        # Get frequency bins
        freqs = np.fft.fftfreq(fft_length, d=1/fps)
        dominant_frequencies[layer_id] = {}
        for filter_id in range(num_filters):
            # Get magnitudes of FFT for each filter
            filter_fft = layer_fft[filter_id]
            # Get the frequency with the highest magnitude (skip the DC component)
            max_id = np.argmax(np.abs(filter_fft[1:])) + 1
            # Store the actual frequency (Hz)
            dominant_frequencies[layer_id][filter_id] = abs(freqs[max_id])
    return dominant_frequencies

def save_dominant_frequencies_to_csv(dominant_frequencies, output_csv_path, image_path):
    file_exists = os.path.exists(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write an empty row if desired
        writer.writerow([])
        # Then write the header row for the table data
        if not file_exists:
            writer.writerow(["image",'Layer ID', 'Filter ID', 'Dominant Frequency'])
        
        for layer_id in sorted(dominant_frequencies.keys()):
            filters = dominant_frequencies[layer_id]
            for filter_id in sorted(filters.keys()):
                writer.writerow([ image_path,layer_id, filter_id, f"{filters[filter_id]:.2f}"])
    print(f"Dominant frequencies saved to '{output_csv_path}'")

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





images_folder = "imgs"  # Update this with the correct folder path

# List all image files (adjust extensions as needed)
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png','.JPG'))]

# Assuming gif_paths is a dictionary with color formats as keys, e.g.:
gif_paths = {
    "RGB": "dummy",  # These values will be modified per image below
}



for image_file in image_files:
    # Construct full path to the image
    image_path = os.path.join(images_folder, image_file)
    # Extract base name without extension to use in output names
    base_name, _ = os.path.splitext(image_file)
    print(f"\nProcessing image: {image_path}")
    for color_format, gif_path in gif_paths.items():

         # Generate unique paths for each image and color format.
        gif_path_modified = f"{base_name}_{color_format.lower()}.gif"
        frames_dir = f"frames_{base_name}_{color_format.lower()}"
        activations_output_dir = f'activations_output_{base_name}_{color_format.lower()}'
        plots_output_dir = f'plots_output_{base_name}_{color_format.lower()}'
        output_csv_path = f'dominant_frequencies_{base_name}_{color_format.lower()}.csv'


        if not os.path.exists(gif_path_modified):
            print(f"Generating flicker image and saving as GIF ({color_format})...")
            frames = flicker_image_and_save_gif(output_gif=gif_path_modified, image_path=image_path, frequency=7, duration=2, fps=30, color_format=color_format)
            # Save frames as images
            save_frames(frames, frames_dir)
            print(f"GIF saved as '{gif_path_modified}'.")
        else:
            print(f"GIF '{gif_path_modified}' already exists. Loading frames from '{frames_dir}'...")
            frames = load_frames(frames_dir)

        # Check if activations directory exists
        if not os.path.exists(activations_output_dir):
            # Perform activations for each color format
            activation_model = ActivationModel(resnet18)
            activations = perform_activations(activation_model, frames, preprocess_seqn)
            save_activations(activations=activations, output_dir=activations_output_dir)
            print(f"Activations saved in '{activations_output_dir}' directory.")
            
        else:
            print(f"Activations directory '{activations_output_dir}' already exists. Skipping activation extraction.")
            activations = load_activations(activations_output_dir)
        ## lets not plot activations as a defult!
        # if not os.path.exists(plots_output_dir):
        #     os.makedirs(plots_output_dir)
        #     plot_activations(activations, plots_output_dir)

        # Perform Fourier Transform on activations
        fourier_transformed_activations = perform_fourier_transform(activations)
        print(f"Fourier Transform performed on activations for {color_format} color format.")

        # Find dominant frequencies
        dominant_frequencies = find_dominant_frequencies(fourier_transformed_activations)
        
        # Save dominant frequencies to CSV
        output_csv_path = f'dominant_frequencies_{color_format.lower()}.csv'
        save_dominant_frequencies_to_csv(dominant_frequencies, output_csv_path,image_path)