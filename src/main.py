from datetime import datetime
import os
from random import shuffle
import torch
import torchvision.transforms as transforms
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.flicker_image import flicker_image_hh_and_save_gif, save_frames, load_frames
from src.core.model import load_activations, init_model, perform_activations
from src.core.signal_processing import perform_fourier_transform, find_dominant_frequencies, save_fft_results_to_db
from src.analysis.frequency_similarity import calculate_frequency_similarity_score
import numpy as np
from src.database import db_stats

# Configuration Constants
# -----------------------
# Image processing settings
FPS = 60
GIF_FREQUENCY1 = 5
GIF_FREQUENCY2 = 6
GIF_DURATION = 10
COLOR_FORMAT = "RGB"

# Color formats to process (format_name: file_extension)
COLOR_FORMATS = {"RGB": ".rgb"}

# Paths and directories
IMAGES_FOLDER = "data/raw"
RESULTS_DIR = "results/exports"
PLOTS_DIR = "results/plots"
DATA_PROCESSED_DIR = "data/processed"

# Processing limits
LIMIT = 10  # Maximum number of images to process

# FFT Analysis settings
REDUCTION_METHOD = "power"  # Options: mean, sum, max, min, median, std, power
THRESHOLD_FACTOR = 1.5
NUM_PEAKS = 3
MIN_SNR = 3.0
PEAK_DETECTION_METHOD = "two_neighbours"  # Options: two_neighbours, four_neighbours, snr
SIMILARITY_TOLERANCE = 0.5

# Database settings
SAVE_TO_DB = True  # Whether to save results to database

# Define preprocessing transformations
preprocess_seqn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    """Main function to process images and analyze activations."""
    # Get current timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting processing at {timestamp}")

    # Initialize the stats table
    filter_stats_table = {}  # {(layer_id, filter_id): {"total_similarity_score": 0.0, "total_images": 0}}

    # Initialize model
    activation_model = init_model()

    # List all image files (adjust extensions as needed)
    image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Shuffle and limit the number of images to process
    shuffle(image_files)
    image_files = image_files[:LIMIT]

    # Use the COLOR_FORMATS constant defined at the top of the file

    # Process each image
    for counter, image_file in enumerate(image_files):
        if counter >= LIMIT:
            break

        print(f"\nProcessing image {counter+1}/{len(image_files)}: {image_file}")
        image_path = os.path.join(IMAGES_FOLDER, image_file)
        base_name, _ = os.path.splitext(image_file)

        for color_format, _ in COLOR_FORMATS.items():
            # Generate unique paths for each image and color format
            gif_path = f"{RESULTS_DIR}/{base_name}_{color_format.lower()}.gif"
            frames_dir = f"{DATA_PROCESSED_DIR}/frames_{base_name}_{color_format.lower()}"
            activations_dir = f'{DATA_PROCESSED_DIR}/activations_{base_name}_{color_format.lower()}'

            # Generate or load frames
            if not os.path.exists(gif_path):
                print(f"Generating flicker image ({color_format})...")
                frames = flicker_image_hh_and_save_gif(
                    image_path=image_path,
                    output_gif=gif_path,
                    duration=GIF_DURATION,
                    frequency1=GIF_FREQUENCY1,
                    frequency2=GIF_FREQUENCY2,
                    fps=FPS,
                    color_format=color_format
                )
                # save_frames(frames, frames_dir)
            else:
                print(f"Loading existing frames from {frames_dir}...")
                frames = load_frames(frames_dir)

            # Generate or load activations
            if not os.path.exists(activations_dir):
                print("Extracting activations...")
                activations = perform_activations(activation_model, frames, preprocess_seqn)
            else:
                print(f"Loading existing activations from {activations_dir}...")
                activations = load_activations(activations_dir)

            # Perform Fourier Transform on activations
            print(f"Performing FFT analysis using {REDUCTION_METHOD} reduction method...")
            fourier_transformed_activations = perform_fourier_transform(activations, reduction_method=REDUCTION_METHOD)

            # Find dominant frequencies
            print("Finding dominant frequencies...")
            dominant_frequencies = find_dominant_frequencies(
                fourier_transformed_activations,
                fps=FPS,
                threshold_factor=THRESHOLD_FACTOR,
                num_peaks=NUM_PEAKS,
                min_snr=MIN_SNR,
                method=PEAK_DETECTION_METHOD
            )

            # Save FFT results to database
            print("Saving FFT results to database...")
            run_id = save_fft_results_to_db(
                image_path=image_path,
                fourier_transformed_activations=fourier_transformed_activations,
                dominant_frequencies=dominant_frequencies,
                color_format=color_format,
                fps=FPS,
                reduction_method=REDUCTION_METHOD,
                gif_frequency1=GIF_FREQUENCY1,
                gif_frequency2=GIF_FREQUENCY2,
            )
            print(f"FFT results saved with run ID: {run_id}")

            print("Calculating filter statistics...")
            for layer_id in dominant_frequencies:
                for filter_id in dominant_frequencies[layer_id]:
                    # Get the full FFT data for this filter
                    fft_vals = fourier_transformed_activations[layer_id][filter_id]
                    magnitudes = np.abs(fft_vals)

                    # Generate frequency bins
                    fft_length = len(magnitudes)
                    freqs = np.fft.fftfreq(fft_length, d=1/FPS)

                    # Get only positive frequencies
                    positive_mask = freqs > 0
                    positive_freqs = freqs[positive_mask]
                    positive_magnitudes = magnitudes[positive_mask]

                    # Calculate similarity score
                    similarity_score, _ = calculate_frequency_similarity_score(
                        frequencies=positive_freqs,
                        magnitudes=positive_magnitudes,
                        target_freq1=GIF_FREQUENCY1,
                        target_freq2=GIF_FREQUENCY2,
                        tolerance=SIMILARITY_TOLERANCE
                    )

                    # Update statistics
                    filter_key = (layer_id, filter_id)
                    if filter_key not in filter_stats_table:
                        filter_stats_table[filter_key] = {
                            "total_similarity_score": 0.0,
                            "total_images": 0
                        }

                    filter_stats_table[filter_key]["total_similarity_score"] += similarity_score
                    filter_stats_table[filter_key]["total_images"] += 1

            # Save filter statistics to database
            print("Saving filter statistics to database...")
            db_stats.update_filter_stats(filter_stats_table, image_path)

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
