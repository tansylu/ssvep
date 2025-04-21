from datetime import datetime
import os
from random import shuffle
import torch
import torchvision.transforms as transforms
from flicker_image import flicker_image_hh_and_save_gif #,flicker_image_and_save_gif  // if we want to flicker the image as whole 
from model import  get_activations, load_activations, save_activations, plot_activations,init_model, reduce_activation
from signal_processing import perform_fourier_transform, find_dominant_frequencies, save_dominant_frequencies_to_csv
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


# Define preprocessing transformations
print("Creating preprocessing sequence...")
preprocess_seqn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

images_folder = "imgs"

# List all image files (adjust extensions as needed)
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png','.JPG'))]

# Assuming gif_paths is a dictionary with color formats as keys, e.g.:
gif_paths = {   
    "RGB": ".rgb",
    #"HSV": ".hsv" # These values will be modified per image below
}

def plot_and_save_spectrums(fourier_transformed_activations, output_dir, fps, dominant_frequencies, gif_frequency1,gif_frequency2):
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
            harmonic_tolerance = 0.1
            harmonics_freq1 = [n * gif_frequency1 for n in range(1, 11)]
            harmonics_freq2 = [n * gif_frequency2 for n in range(1, 11)]

            is_harmonic = any(abs(dominant_frequency - h) < harmonic_tolerance for h in harmonics_freq1 + harmonics_freq2)
            
            if not is_harmonic:
                plt.figure(figsize=(10, 5))
                # Exclude the DC component by starting from index 1
                plt.bar(freqs[1:], np.abs(layer_fft[filter_id][1:]), width=0.05, label=f'Filter {filter_id}')
                plt.title(f'Layer {layer_id+1} Filter {filter_id} Spectrum')
                plt.xlabel('Frequency')
                plt.ylabel('Magnitude')
                plt.legend()
                # Add ticks at the target frequency and its harmonics
                harmonic_ticks1 = [n * gif_frequency1 for n in range(-2, 3)]
                harmonic_ticks2 = [n * gif_frequency2 for n in range(-2, 3)]
                for tick in harmonic_ticks1:
                    plt.axvline(x=tick, color='r', linestyle='--', linewidth=0.5, label='f1 harmonic' if tick == gif_frequency1 else "")
                for tick in harmonic_ticks2:
                    plt.axvline(x=tick, color='g', linestyle='--', linewidth=0.5, label='f2 harmonic' if tick == gif_frequency2 else "")

                plot_path = os.path.join(output_dir, f'layer_{layer_id}_filter_{filter_id}_spectrum.png')
                plt.savefig(plot_path)
                plt.close()
                print(f'Saved spectrum plot for Layer {layer_id+1} Filter {filter_id} at {plot_path}')

timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
output_csv_path = f'dominant_frequencies.csv'
resnet18 = init_model() # Initialize only once

LIMIT = 10000
COUNTER = 0
# shuffle the list of image files
shuffle(image_files)

# Update the main processing loop to pass the dominant frequencies and gif_frequency to the plot_and_save_spectrums function
for image_file in image_files:
    if COUNTER >= LIMIT:
        print(f"Processed {LIMIT} images. Stopping further processing.")
        break
    print(f"\nProcessing image {COUNTER + 1}/{LIMIT}: {image_file}")
    COUNTER += 1
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
        output_csv_path_2n = f'dominant_frequencies_2n.csv'
        output_csv_path_4n = f'dominant_frequencies_4n.csv'
        output_csv_path_snr = f'dominant_frequencies_snr.csv'

        if not os.path.exists(gif_path_modified):
            # print(f"Generating flicker image and saving as GIF ({color_format})...")
            frames = flicker_image_hh_and_save_gif(image_path=image_path, output_gif=gif_path_modified, duration=5, frequency1=5,frequency2=6 ,fps=24,color_format=color_format)
            # Save frames as images
            # save_frames(frames, frames_dir)
            # print(f"GIF saved as '{gif_path_modified}'.")
        else:
            print(f"GIF '{gif_path_modified}' already exists. Loading frames from '{frames_dir}'...")
            # frames = load_frames(frames_dir)

        # Check if activations directory exists
        if not os.path.exists(activations_output_dir):
            # Perform activations for each color format
            activation_model = init_model()
            activations = perform_activations(activation_model, frames, preprocess_seqn)
            # save_activations(activations=activations, output_dir=activations_output_dir)
            # print(f"Activations saved in '{activations_output_dir}' directory.")
        else:
            print(f"Activations directory '{activations_output_dir}' already exists. Skipping activation extraction.")
            activations = load_activations(activations_output_dir)

        # Perform Fourier Transform on activations
        fourier_transformed_activations = perform_fourier_transform(activations, reduction_method='median')
        # print(f"Fourier Transform performed on activations for {color_format} color format.")

        # Find dominant frequencies
        dominant_frequencies_2n = find_dominant_frequencies(fourier_transformed_activations, fps=24, threshold_factor=1.5, num_peaks=3, min_snr=3.0, method='two_neighbours')
        dominant_frequencies_4n = find_dominant_frequencies(fourier_transformed_activations, fps=24, threshold_factor=1.2, num_peaks=3, min_snr=3.0, method='four_neighbours')
        dominant_frequencies_snr = find_dominant_frequencies(fourier_transformed_activations, fps=24, threshold_factor=2.0, num_peaks=3, min_snr=2.0, method='snr')

        # Save dominant frequencies to CSV
        save_dominant_frequencies_to_csv(dominant_frequencies_2n, output_csv_path_2n, image_path, gif_frequency1=5,gif_frequency2=6)
        save_dominant_frequencies_to_csv(dominant_frequencies_4n, output_csv_path_4n, image_path, gif_frequency1=5,gif_frequency2=6)
        save_dominant_frequencies_to_csv(dominant_frequencies_snr, output_csv_path_snr, image_path, gif_frequency1=5,gif_frequency2=6)
        
        # Plot and save spectrums
        # spectrum_output_dir = f'spectrum_plots_{base_name}_{color_format.lower()}'
        # plot_and_save_spectrums(fourier_transformed_activations, spectrum_output_dir, fps=24, dominant_frequencies=dominant_frequencies, gif_frequency1=5,gif_frequency2=6)
        # print(f"Spectrums plotted and saved in '{spectrum_output_dir}' directory.")