from datetime import datetime
import os
from random import shuffle
import torch
import torchvision.transforms as transforms
from flicker_image import flicker_image_hh_and_save_gif #,flicker_image_and_save_gif  // if we want to flicker the image as whole
from model import  get_activations, load_activations, init_model
from signal_processing import perform_fourier_transform, find_dominant_frequencies, save_fft_results_to_db, is_harmonic_frequency, HarmonicType
from frequency_similarity import calculate_frequency_similarity_score
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import csv
import sys
import argparse
import db  # Import our database module
import db_stats  # Import our database stats module


def save_frames(frames, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_image = Image.fromarray(frame.astype(np.uint8))  # Convert numpy array to PIL Image
        frame_path = os.path.join(frames_dir, f"frame_{i}.png")
        frame_image.save(frame_path)
    #print(f"Frames saved in '{frames_dir}' directory.")

def save_filter_stats_to_csv(filter_stats_table, csv_file_path, image_file=None, db_only=False):
    """
    Save filter statistics to a CSV file.
    Args:
        filter_stats_table: Dictionary with filter statistics
        csv_file_path: Path to the CSV file
        image_file: Current image being processed (optional)
        db_only: If True, skip saving to CSV
    """
    # Skip if db-only mode is enabled
    if db_only:
        return

    # Check if file exists to determine if we need to create a new file
    file_exists = os.path.exists(csv_file_path)

    # Prepare data for CSV
    csv_data = {}

    # If file exists, read the current data
    if file_exists:
        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            try:
                next(reader)  # Skip header row
                for row in reader:
                    if len(row) >= 5:  # Ensure row has enough columns
                        layer_id, filter_id = row[0], row[1]
                        key = (layer_id, filter_id)
                        csv_data[key] = {
                            'layer': layer_id,
                            'filter': filter_id,
                            'total_similarity_score': float(row[2]),
                            'total_images': int(row[3]),
                            'avg_similarity_score': float(row[4])
                        }
            except StopIteration:
                # File is empty or has only a header
                pass

    # Update with current data
    for (layer_id, filter_id), stats in filter_stats_table.items():
        total_images = stats["total_images"]
        if total_images > 0:
            avg_similarity_score = stats["total_similarity_score"] / total_images
            key = (str(layer_id), str(filter_id))
            csv_data[key] = {
                'layer': layer_id,
                'filter': filter_id,
                'total_similarity_score': stats["total_similarity_score"],
                'total_images': total_images,
                'avg_similarity_score': avg_similarity_score
            }

    # Sort by average similarity score (higher means more similar to expected frequencies)
    sorted_data = sorted(csv_data.values(), key=lambda x: x['avg_similarity_score'], reverse=True)

    # Write to CSV
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Layer", "Filter", "Total Similarity Score", "Total Images", "Avg Similarity Score"])

        # Write data
        for item in sorted_data:
            writer.writerow([
                item['layer'],
                item['filter'],
                f"{item['total_similarity_score']:.4f}",
                item['total_images'],
                f"{item['avg_similarity_score']:.4f}"
            ])

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
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

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


            # Check for intermodulation products
            is_intermod = is_harmonic_frequency(
                peak_frequencies=peak_frequencies,
                freq1=gif_frequency1,
                freq2=gif_frequency2,
                harmonic_type=HarmonicType.INTERMOD,
                tolerance=harmonic_tolerance
            )

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
parser.add_argument('--non-intermod', action='store_true', default=True,
                    help='Only plot spectrums that are not intermodulation products (f1*f2)')
parser.add_argument('--export-stats', type=str, help='Export filter statistics to the specified CSV file')
parser.add_argument('--db-only', action='store_true', help='Only save results to database, skip CSV files')

args = parser.parse_args()

# Check if we should export filter statistics from the database
if args.export_stats:
    print(f"Exporting filter statistics to {args.export_stats}...")
    db_stats.export_filter_stats_to_csv(args.export_stats)
    print(f"Filter statistics exported to {args.export_stats}")
    # Exit if we're only exporting stats
    if len(sys.argv) == 3:  # Only --export-stats and the filename were provided
        sys.exit(0)

timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize the stats table
filter_stats_table = {}  # {(layer_id, filter_id): {"different": 0, "same": 0, "total": 0}}
csv_stats_file = f'filter_stats_{timestamp_now}.csv'

# Initialize only once
resnet18 = init_model()

# Set a small limit for testing
LIMIT = 100
COUNTER = 0
# shuffle the list of image files
shuffle(image_files)
# Take only the first few images for testing
image_files = image_files[:LIMIT]
activation_model = init_model()

# Update the main processing loop
for image_file in image_files:
    if COUNTER >= LIMIT:
        break
    COUNTER += 1
    print(f"\nProcessing image: {image_file}")
    image_path = os.path.join(images_folder, image_file)
    base_name, _ = os.path.splitext(image_file)

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
        reduction_method = args.reduction
        fps = 60
        gif_frequency1 = 5
        gif_frequency2 = 6

        fourier_transformed_activations = perform_fourier_transform(activations, reduction_method=reduction_method)
        print(f"Using {reduction_method} reduction method for FFT analysis")
        print(f"Fourier Transform performed on activations for {color_format} color format.")

        # Find dominant frequencies
        dominant_frequencies_2n = find_dominant_frequencies(fourier_transformed_activations, fps=fps, threshold_factor=1.5, num_peaks=3, min_snr=3.0, method='two_neighbours')
        # dominant_frequencies_4n = find_dominant_frequencies(fourier_transformed_activations, fps=24, threshold_factor=1.2, num_peaks=3, min_snr=3.0, method='four_neighbours')
        # dominant_frequencies_snr = find_dominant_frequencies(fourier_transformed_activations, fps=24, threshold_factor=2.0, num_peaks=3, min_snr=2.0, method='snr')

        # Save FFT results to database
        print(f"Saving FFT results to database for {image_file} ({color_format})...")
        run_id = save_fft_results_to_db(
            image_path=image_path,
            fourier_transformed_activations=fourier_transformed_activations,
            dominant_frequencies=dominant_frequencies_2n,
            color_format=color_format,
            fps=fps,
            reduction_method=reduction_method,
            gif_frequency1=gif_frequency1,
            gif_frequency2=gif_frequency2,
        )
        print(f"FFT results saved to database with run ID: {run_id}")

        # Update the filter statistics
        for layer_id in dominant_frequencies_2n:
            for filter_id in dominant_frequencies_2n[layer_id]:
                peak_frequencies = dominant_frequencies_2n[layer_id][filter_id]

                # Get the full FFT data for this filter
                fft_vals = fourier_transformed_activations[layer_id][filter_id]
                magnitudes = np.abs(fft_vals)

                # Generate frequency bins
                fft_length = len(magnitudes)
                freqs = np.fft.fftfreq(fft_length, d=1/fps)

                # Get only positive frequencies
                positive_mask = freqs > 0
                positive_freqs = freqs[positive_mask]
                positive_magnitudes = magnitudes[positive_mask]

                # Calculate similarity score using the full spectrum
                similarity_score, _ = calculate_frequency_similarity_score(
                    frequencies=positive_freqs,
                    magnitudes=positive_magnitudes,
                    target_freq1=gif_frequency1,
                    target_freq2=gif_frequency2,
                    tolerance=0.5  # Use a more reasonable tolerance
                )

                filter_key = (layer_id, filter_id)
                if filter_key not in filter_stats_table:
                    filter_stats_table[filter_key] = {
                        "total_similarity_score": 0.0,
                        "total_images": 0
                    }

                # Add the similarity score to the total
                filter_stats_table[filter_key]["total_similarity_score"] += similarity_score
                filter_stats_table[filter_key]["total_images"] += 1

        # Save filter statistics to CSV file after each image
        save_filter_stats_to_csv(filter_stats_table, csv_stats_file, image_file, args.db_only)

        # Save filter statistics to database
        print(f"Saving filter statistics to database for {image_file}...")
        db_stats.update_filter_stats(filter_stats_table, image_path)
        print("Filter statistics saved to database.")
