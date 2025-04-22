from datetime import datetime
import os
from random import shuffle
import torch
import torchvision.transforms as transforms
from flicker_image import flicker_image_hh_and_save_gif #,flicker_image_and_save_gif  // if we want to flicker the image as whole
from model import  get_activations, load_activations, init_model
from signal_processing import perform_fourier_transform, find_dominant_frequencies, is_harmonic_frequency, HarmonicType
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import csv
import sys
import argparse
import json


def save_frames(frames, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_image = Image.fromarray(frame.astype(np.uint8))  # Convert numpy array to PIL Image
        frame_path = os.path.join(frames_dir, f"frame_{i}.png")
        frame_image.save(frame_path)
    #print(f"Frames saved in '{frames_dir}' directory.")

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
#print("Creating preprocessing sequence...")
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
    "RGB": ".rgb"
}

def plot_and_save_spectrums(fourier_transformed_activations, output_dir, fps, dominant_frequencies, gif_frequency1, gif_frequency2, specific_filter_id=None, specific_layer_id=None, non_harmonic_f1=False, non_harmonic_any=False, non_intermod=False):
    """
    Plots and saves the spectrums of the Fourier Transformed activations.
    Args:
        fourier_transformed_activations: {layer_id: np.array(num_filters, fft_length)}
        output_dir: The directory where the plots will be saved.
        dominant_frequencies: {layer_id: {filter_id: dominant_frequency}}
        gif_frequency1: The first frequency of the GIF used for comparison.
        gif_frequency2: The second frequency of the GIF used for comparison.
        specific_filter_id: If provided, only plot this specific filter ID.
        specific_layer_id: If provided, only plot this specific layer ID.
        non_harmonic_f1: If True, only plot spectrums that are not harmonics of frequency 1.
        non_harmonic_any: If True, only plot spectrums that are not harmonics of either frequency.
        non_intermod: If True, only plot spectrums that are not intermodulation products (f1*f2).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If specific layer ID is provided, only process that layer
    layer_ids = [specific_layer_id] if specific_layer_id is not None else fourier_transformed_activations.keys()

    for layer_id in layer_ids:
        # Skip if the layer ID doesn't exist in the data
        if layer_id not in fourier_transformed_activations:
            #print(f"Layer ID {layer_id} not found in the data.")
            continue

        layer_fft = fourier_transformed_activations[layer_id]
        num_filters, fft_length = layer_fft.shape

        # Generate frequency bins (without fftshift)
        freqs = np.fft.fftfreq(fft_length, d=1/fps)  # freq in Hz

        # Get only positive frequencies (excluding zero/DC)
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]

        # If specific filter ID is provided, only process that filter
        filter_ids = [specific_filter_id] if specific_filter_id is not None and specific_filter_id < num_filters else range(num_filters)

        for filter_id in filter_ids:
            peak_frequencies = dominant_frequencies[layer_id][filter_id]
            # Set harmonic detection parameters
            harmonic_tolerance = 1

            # Check for different types of harmonics using the global method
            is_harmonic = is_harmonic_frequency(
                peak_frequencies=peak_frequencies,
                freq1=gif_frequency1,
                freq2=gif_frequency2,
                harmonic_type=HarmonicType.ANY,
                tolerance=harmonic_tolerance
            )

            # Check specifically for harmonics of frequency 1
            is_harmonic_f1 = is_harmonic_frequency(
                peak_frequencies=peak_frequencies,
                freq1=gif_frequency1,
                freq2=gif_frequency2,
                harmonic_type=HarmonicType.FREQ1,
                tolerance=harmonic_tolerance
            )

            # Check specifically for harmonics of frequency 2
            is_harmonic_f2 = is_harmonic_frequency(
                peak_frequencies=peak_frequencies,
                freq1=gif_frequency1,
                freq2=gif_frequency2,
                harmonic_type=HarmonicType.FREQ2,
                tolerance=harmonic_tolerance
            )

            # Check for intermodulation products
            is_intermod = is_harmonic_frequency(
                peak_frequencies=peak_frequencies,
                freq1=gif_frequency1,
                freq2=gif_frequency2,
                harmonic_type=HarmonicType.INTERMOD,
                tolerance=harmonic_tolerance
            )

            # Apply filters based on command-line arguments
            if non_harmonic_f1 and is_harmonic_f1:
                continue

            if non_harmonic_any and (is_harmonic_f1 or is_harmonic_f2):
                continue

            if non_intermod and is_intermod:
                continue
            plt.figure(figsize=(10, 5))
            # Get the positive frequency components of the FFT
            positive_fft = np.abs(layer_fft[filter_id][positive_mask])

            # Filter frequencies to only show up to 35 Hz
            freq_mask = positive_freqs <= 35
            plot_freqs = positive_freqs[freq_mask]
            plot_fft = positive_fft[freq_mask]

            # Plot only positive frequencies up to 35 Hz
            plt.bar(plot_freqs, plot_fft, width=0.05, label=f'Filter {filter_id}')
            plt.title(f'Layer {layer_id+1} Filter {filter_id} Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.legend()

            # Add more ticks on x-axis
            # Create ticks every 1 Hz up to 35 Hz
            x_ticks = np.arange(0, 36, 1)
            plt.xticks(x_ticks)
            plt.grid(axis='x', linestyle='--', alpha=0.7)

            # Set x-axis limit to 35 Hz
            plt.xlim(0, 16)

            # Generate harmonic ticks for visualization
            max_harmonic = 12  # Show up to 11th harmonic
            harmonic_ticks1 = [n * gif_frequency1 for n in range(1, max_harmonic)]  # Start from 1 to skip zero
            harmonic_ticks2 = [n * gif_frequency2 for n in range(1, max_harmonic)]  # Start from 1 to skip zero

            for tick in harmonic_ticks1:
                plt.axvline(x=tick, color='r', linestyle='--', linewidth=0.5, label='f1 harmonic' if tick == gif_frequency1 else "")

            for tick in harmonic_ticks2:
                plt.axvline(x=tick, color='g', linestyle='--', linewidth=0.5, label='f2 harmonic' if tick == gif_frequency2 else "")

            plot_path = os.path.join(output_dir, f'layer_{layer_id}_filter_{filter_id}_spectrum.png')
            plt.savefig(plot_path)
            plt.close()
            #print(f'Saved spectrum plot for Layer {layer_id+1} Filter {filter_id} at {plot_path}')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process images and plot activation spectrums.')
parser.add_argument('--filter-id', type=int, help='Specific filter ID to plot')
parser.add_argument('--layer-id', type=int, help='Specific layer ID to plot')
parser.add_argument('--reduction', type=str, default='power',
                    choices=['mean', 'sum', 'max', 'min', 'median', 'std', 'power'],
                    help='Reduction method for spatial dimensions (default: power)')
parser.add_argument('--non-harmonic-f1', action='store_true',
                    help='Only plot spectrums that are not harmonics of frequency 1')
parser.add_argument('--non-harmonic-any', action='store_true',
                    help='Only plot spectrums that are not harmonics of either frequency')
parser.add_argument('--non-intermod', action='store_true',
                    help='Only plot spectrums that are not intermodulation products (f1*f2)')

args = parser.parse_args()

timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize only once
resnet18 = init_model()

# Create a counter dictionary to track Different/Same counts per layer+filter combination
filter_counts = {}  # {(layer_id, filter_id): {"Different": count, "Same": count, "images": {image_path: "Different"|"Same"}}}

# Output JSON file path
filter_counts_json = f'filter_counts_{timestamp_now}.json'

# Set a small limit for testing
LIMIT = 2
COUNTER = 0
# shuffle the list of image files
shuffle(image_files)
# Take only the first few images for testing
image_files = image_files[:LIMIT]
activation_model = init_model()

# Update the main processing loop to pass the dominant frequencies and gif_frequency to the plot_and_save_spectrums function
for image_file in image_files:
    if COUNTER >= LIMIT:
        #print(f"Processed {LIMIT} images. Stopping further processing.")
        break
    #print(f"\nProcessing image {COUNTER + 1}/{LIMIT}: {image_file}")
    COUNTER += 1
    print(f"\nProcessing image: {image_file}")
    # Construct full path to the image
    image_path = os.path.join(images_folder, image_file)
    # Extract base name without extension to use in output names
    base_name, _ = os.path.splitext(image_file)
    #print(f"\nProcessing image: {image_path}")
    for color_format, gif_path in gif_paths.items():

        # Generate unique paths for each image and color format.
        gif_path_modified = f"{base_name}_{color_format.lower()}.gif"
        frames_dir = f"frames_{base_name}_{color_format.lower()}"
        activations_output_dir = f'activations_output_{base_name}_{color_format.lower()}'
        plots_output_dir = f'plots_output_{base_name}_{color_format.lower()}'

        if not os.path.exists(gif_path_modified):
            # #print(f"Generating flicker image and saving as GIF ({color_format})...")
            frames = flicker_image_hh_and_save_gif(image_path=image_path, output_gif=gif_path_modified, duration=10, frequency1=5,frequency2=6 ,fps=60,color_format=color_format)
            # Save frames as images
            # save_frames(frames, frames_dir)
            # #print(f"GIF saved as '{gif_path_modified}'.")
        else:
            #print(f"GIF '{gif_path_modified}' already exists. Loading frames from '{frames_dir}'...")
            frames = load_frames(frames_dir)

        # Check if activations directory exists
        if not os.path.exists(activations_output_dir):
            # Perform activations for each color format

            activations = perform_activations(activation_model, frames, preprocess_seqn)
            # save_activations(activations=activations, output_dir=activations_output_dir)
            # #print(f"Activations saved in '{activations_output_dir}' directory.")
        else:
            #print(f"Activations directory '{activations_output_dir}' already exists. Skipping activation extraction.")
            activations = load_activations(activations_output_dir)

        # Perform Fourier Transform on activations
        fourier_transformed_activations = perform_fourier_transform(activations, reduction_method=args.reduction)
        #print(f"Using {args.reduction} reduction method for FFT analysis")
        # #print(f"Fourier Transform performed on activations for {color_format} color format.")

        # Find dominant frequencies
        dominant_frequencies_2n = find_dominant_frequencies(fourier_transformed_activations, fps=60, threshold_factor=1.5, num_peaks=3, min_snr=3.0, method='two_neighbours')
        # dominant_frequencies_4n = find_dominant_frequencies(fourier_transformed_activations, fps=24, threshold_factor=1.2, num_peaks=3, min_snr=3.0, method='four_neighbours')
        # dominant_frequencies_snr = find_dominant_frequencies(fourier_transformed_activations, fps=24, threshold_factor=2.0, num_peaks=3, min_snr=2.0, method='snr')

        # Instead of updating CSV, we'll update our counter dictionary

        # Analyze the dominant frequencies before plotting
        #print("\nAnalyzing dominant frequencies for this image...")

        # Count 'Different' vs 'Same' flags
        different_count = 0
        same_count = 0
        different_by_layer = {}
        different_filters = []
        top_different_filters = []  # Initialize this variable for later use

        for layer_id in dominant_frequencies_2n:
            for filter_id in dominant_frequencies_2n[layer_id]:
                peak_frequencies = dominant_frequencies_2n[layer_id][filter_id]

                # Check if any peak is a harmonic
                is_harmonic = is_harmonic_frequency(
                    peak_frequencies=peak_frequencies,
                    freq1=5,  # gif_frequency1
                    freq2=6,  # gif_frequency2
                    harmonic_type=HarmonicType.ANY,
                    tolerance=0.1
                )

                # Create a key for this layer+filter combination
                filter_key = (layer_id, filter_id)

                # Initialize the counter for this filter if it doesn't exist
                if filter_key not in filter_counts:
                    filter_counts[filter_key] = {
                        "Different": 0,
                        "Same": 0,
                        "images": {}
                    }

                # Update the counter
                if is_harmonic:
                    same_count += 1
                    filter_counts[filter_key]["Same"] += 1
                    filter_counts[filter_key]["images"][image_path] = "Same"
                else:
                    different_count += 1
                    filter_counts[filter_key]["Different"] += 1
                    filter_counts[filter_key]["images"][image_path] = "Different"
                    if layer_id not in different_by_layer:
                        different_by_layer[layer_id] = 0
                    different_by_layer[layer_id] += 1
                    different_filters.append((layer_id, filter_id))

        # Calculate percentages
        total_count = different_count + same_count
        different_percent = (different_count / total_count) * 100 if total_count > 0 else 0

        #print(f"\nFlag distribution:")
        #print(f"- Different: {different_count} ({different_percent:.2f}%)")
        #print(f"- Same: {same_count} ({100 - different_percent:.2f}%)")

        if different_count > 0:
            #print("\nDistribution of 'Different' flags by layer:")
            for layer_id in sorted(different_by_layer.keys()):
                layer_percent = (different_by_layer[layer_id] / different_count) * 100
                #print(f"- Layer {layer_id}: {different_by_layer[layer_id]} ({layer_percent:.2f}%)")

            # Calculate the percentage of 'Different' flags per filter
            filter_stats = {}
            total_filters = {}

            # Count total occurrences of each filter
            for layer_id in dominant_frequencies_2n:
                for filter_id in dominant_frequencies_2n[layer_id]:
                    filter_key = (layer_id, filter_id)
                    if filter_key not in total_filters:
                        total_filters[filter_key] = 0
                    total_filters[filter_key] += 1

            # Calculate percentage of 'Different' flags for each filter
            from collections import Counter
            different_filter_counts = Counter(different_filters)
            for filter_key, diff_count in different_filter_counts.items():
                total_count = total_filters.get(filter_key, 0)
                if total_count > 0:
                    percentage = (diff_count / total_count) * 100
                    filter_stats[filter_key] = (diff_count, total_count, percentage)

            # Sort filters by percentage of 'Different' flags
            top_different_filters = sorted(filter_stats.items(), key=lambda x: x[1][2], reverse=True)[:10]

            #print("\nTop 10 filters with highest percentage of 'Different' flags:")
            for (layer_id, filter_id), (diff_count, total_count, percentage) in top_different_filters:
                print(f"- Layer {layer_id}, Filter {filter_id}: {diff_count}/{total_count} ({percentage:.2f}%)")


        # Plot and save spectrums
        spectrum_output_dir = f'spectrum_plots_{base_name}_{color_format.lower()}'

        # If user specified filter/layer IDs or used filtering options, use those
        if args.filter_id is not None or args.layer_id is not None or args.non_harmonic_f1 or args.non_harmonic_any or args.non_intermod:
            plot_and_save_spectrums(
                fourier_transformed_activations,
                spectrum_output_dir,
                fps=60,
                dominant_frequencies=dominant_frequencies_2n,
                gif_frequency1=5,
                gif_frequency2=6,
                specific_filter_id=args.filter_id,
                specific_layer_id=args.layer_id,
                non_harmonic_f1=args.non_harmonic_f1,
                non_harmonic_any=args.non_harmonic_any,
                non_intermod=args.non_intermod
            )
            #print(f"Spectrums plotted and saved in '{spectrum_output_dir}' directory.")

        else:
            plot_and_save_spectrums(
                fourier_transformed_activations,
                spectrum_output_dir,
                fps=60,
                dominant_frequencies=dominant_frequencies_2n,
                gif_frequency1=5,
                gif_frequency2=6,
                specific_filter_id=args.filter_id,
                specific_layer_id=args.layer_id,
                non_harmonic_f1=args.non_harmonic_f1,
                non_harmonic_any=args.non_harmonic_any,
                non_intermod=args.non_intermod
            )
            #print(f"Spectrums plotted and saved in '{spectrum_output_dir}' directory.")

# After processing all images, save the filter counts to a JSON file
# Convert tuple keys to strings for JSON serialization
filter_counts_serializable = {}
for (layer_id, filter_id), counts in filter_counts.items():
    filter_counts_serializable[f"layer_{layer_id}_filter_{filter_id}"] = counts

# Calculate percentages for each filter
for filter_key, counts in filter_counts_serializable.items():
    total = counts["Different"] + counts["Same"]
    if total > 0:
        counts["different_percent"] = (counts["Different"] / total) * 100
        counts["same_percent"] = (counts["Same"] / total) * 100
    else:
        counts["different_percent"] = 0
        counts["same_percent"] = 0

# Save to JSON file
with open(filter_counts_json, 'w') as f:
    json.dump(filter_counts_serializable, f, indent=2)

print(f"\nFilter counts saved to {filter_counts_json}")

# Print summary of results
print("\nSummary of results:")
print(f"Total layer+filter combinations: {len(filter_counts)}")

# Sort filters by percentage of 'Different' flags
sorted_filters = sorted(
    [(k, v) for k, v in filter_counts_serializable.items() if v["Different"] + v["Same"] > 0],
    key=lambda x: x[1]["different_percent"],
    reverse=True
)

# Print top 10 filters with highest percentage of 'Different' flags
print("\nTop 10 filters with highest percentage of 'Different' flags:")
for i, (filter_key, counts) in enumerate(sorted_filters[:10], 1):
    print(f"{i}. {filter_key}: {counts['Different']}/{counts['Different'] + counts['Same']} ({counts['different_percent']:.2f}%)")
