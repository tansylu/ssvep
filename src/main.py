from datetime import datetime
import os
import torch
import torchvision.transforms as transforms
from flicker_image import flicker_image_and_save_gif
from model import ActivationModel, get_activations, load_activations, save_activations, plot_activations,init_model
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

def perform_fourier_transform(activations, reduction_method='mean'):
    """
    Performs FFT on activation time series for each layer and filter.
    Args:
        activations: {layer_id: [frame1_tensor(1,filters,height,width), frame2_tensor...]}
        reduction_method: Method to reduce spatial dimensions ('mean', 'sum', 'max', 'min', 'median')
    Returns:
        {layer_id: numpy_array(num_filters, fft_length)}
    """
    reduction_methods = {
        'mean': np.mean,
        'sum': np.sum,
        'max': np.max,
        'min': np.min,
        'median': np.median
    }
    
    if reduction_method not in reduction_methods:
        raise ValueError(f"Invalid reduction method: {reduction_method}. Choose from 'mean', 'sum', 'max', 'min', 'median'.")#try l2, better csv plots,
    
    reduce_fn = reduction_methods[reduction_method]
    
    fourier_transformed_activations = {}
    for layer_id, frames in activations.items():
        num_filters = frames[0].shape[1]
        num_frames = len(frames)
        
        # Initialize the Fourier transformed activations array
        fourier_transformed_activations[layer_id] = np.zeros((num_filters, num_frames))
        
        # Iterate over each filter
        for filter_id in range(num_filters):
            # Extract the temporal sequence for each filter using the specified reduction method
            temporal_sequence = [reduce_fn(frame[0, filter_id, :, :]) for frame in frames]
            
            # Perform Fourier Transform on the temporal sequence
            fourier_transformed_activations[layer_id][filter_id] = np.abs(fft(temporal_sequence))
    
    return fourier_transformed_activations

def find_dominant_frequencies(fourier_transformed_activations, fps):
    """
    Args:
        fourier_transformed_activations: {layer_id: np.array(num_filters, fft_length)}
        fps: sampling rate in Hz (frames per second)
    Returns:
        {layer_id: {filter_id: dominant_frequency}}
    """
    dominant_frequencies = {}
    for layer_id, layer_fft in fourier_transformed_activations.items():
        num_filters, fft_length = layer_fft.shape
        
        # Get frequency bins (convert to Hz by multiplying by fps)
        freqs = np.fft.fftfreq(fft_length, d=1/fps)  # freq in Hz
        freqs = np.fft.fftshift(freqs)  # Shift the zero frequency to the center
        
        dominant_frequencies[layer_id] = {}
        for filter_id in range(num_filters):
            # Get magnitudes of FFT for each filter
            filter_fft = layer_fft[filter_id]
            # Get the frequency with the highest magnitude (skip the DC component)
            # Ensure the FFT is shifted to avoid the DC component causing issues
            max_id = np.argmax(np.abs(filter_fft[1:])) + 1
            # Store the actual frequency (Hz) corresponding to the peak of the FFT
            dominant_frequencies[layer_id][filter_id] = abs(freqs[max_id])
    return dominant_frequencies


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
    "HSV": "dummy",  # These values will be modified per image below
}

def save_dominant_frequencies_to_csv(dominant_frequencies, output_csv_path, image_path, gif_frequency):
    file_exists = os.path.exists(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write an empty row if desired
        writer.writerow([])
        # Then write the header row for the table data
        if not file_exists:
            writer.writerow(["Image", "Layer ID", "Filter ID", "Dominant Frequency", "GIF Frequency", "Difference", "Flag"])
        for layer_id in sorted(dominant_frequencies.keys()):
            filters = dominant_frequencies[layer_id]
            for filter_id in sorted(filters.keys()):
                dominant_frequency = filters[filter_id]
                difference = abs(dominant_frequency - gif_frequency)
                # Check if the dominant frequency is a harmonic of the GIF frequency
                is_harmonic = any(abs(dominant_frequency - n * gif_frequency) < 0.1 for n in range(1, 11))  # Adjust range as needed
                flag = "Different" if not is_harmonic else "Same"
                writer.writerow([image_path, layer_id, filter_id, f"{dominant_frequency:.2f}", gif_frequency, f"{difference:.2f}", flag])
        print(f"Dominant frequencies saved to '{output_csv_path}'")

def plot_and_save_spectrums(fourier_transformed_activations, output_dir, fps, dominant_frequencies, gif_frequency):
    """
    Plots and saves the spectrums of the Fourier Transformed activations for filters marked as "Different".
    Args:
        fourier_transformed_activations: {layer_id: np.array(num_filters, fft_length)}
        output_dir: The directory where the plots will be saved.
        dominant_frequencies: {layer_id: {filter_id: dominant_frequency}}
        gif_frequency: The frequency of the GIF used for comparison.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for layer_id, layer_fft in fourier_transformed_activations.items():
        num_filters, fft_length = layer_fft.shape
        freqs = np.fft.fftfreq(fft_length, d=1/fps)  # freq in Hz
        freqs = np.fft.fftshift(freqs)  # Shift the zero frequency to the center
        
        for filter_id in range(num_filters):
            dominant_frequency = dominant_frequencies[layer_id][filter_id]
            # Check if the dominant frequency is a harmonic of the GIF frequency
            is_harmonic = any(abs(dominant_frequency - n * gif_frequency) < 0.1 for n in range(1, 11))
            if not is_harmonic:
                plt.figure(figsize=(10, 5))
                # Exclude the DC component by starting from index 1
                plt.bar(freqs[1:], np.abs(layer_fft[filter_id][1:]), width=0.05, label=f'Filter {filter_id}')
                plt.title(f'Layer {layer_id+1} Filter {filter_id} Spectrum')
                plt.xlabel('Frequency')
                plt.ylabel('Magnitude')
                plt.legend()
                # Add ticks at the target frequency and its harmonics
                harmonic_ticks = [n * gif_frequency for n in range(-2, 2)]
                for tick in harmonic_ticks:
                    plt.axvline(x=tick, color='r', linestyle='--', linewidth=0.5)
                plot_path = os.path.join(output_dir, f'layer_{layer_id}_filter_{filter_id}_spectrum.png')
                plt.savefig(plot_path)
                plt.close()
                print(f'Saved spectrum plot for Layer {layer_id+1} Filter {filter_id} at {plot_path}')

timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
output_csv_path = f'dominant_frequencies_{timestamp_now}.csv'
# Update the main processing loop to pass the dominant frequencies and gif_frequency to the plot_and_save_spectrums function
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
            frames = flicker_image_and_save_gif( image_path=image_path, color_format=color_format, output_gif=gif_path_modified, duration=5, frequency=4, fps=24)
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

        # Perform Fourier Transform on activations
        fourier_transformed_activations = perform_fourier_transform(activations, reduction_method='mean')
        print(f"Fourier Transform performed on activations for {color_format} color format.")

        # Find dominant frequencies
        dominant_frequencies = find_dominant_frequencies(fourier_transformed_activations, fps=24)
        
        # Save dominant frequencies to CSV
     
        save_dominant_frequencies_to_csv(dominant_frequencies, output_csv_path, image_path, gif_frequency=4)

        # Plot and save spectrums
        spectrum_output_dir = f'spectrum_plots_{base_name}_{color_format.lower()}'
        plot_and_save_spectrums(fourier_transformed_activations, spectrum_output_dir, fps=24, dominant_frequencies=dominant_frequencies, gif_frequency=4)
        print(f"Spectrums plotted and saved in '{spectrum_output_dir}' directory.")