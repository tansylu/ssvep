#!/usr/bin/env python3
"""
Script to generate spectrum data from the FFT database for a specific filter ID and layer.
This script allows you to visualize the frequency spectrum for a specific filter and layer
across all runs or for specific runs in the database.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.database import db
from datetime import datetime
import imageio
import tempfile

def get_runs_with_filter_layer(filter_id, layer_id):
    """
    Get all runs that have data for the specified filter and layer.

    Args:
        filter_id (int): The filter ID to search for
        layer_id (int): The layer ID to search for

    Returns:
        list: List of run dictionaries containing run information
    """
    conn = db.get_connection()
    cursor = conn.cursor()

    # Query to find all runs that have data for the specified filter and layer
    cursor.execute(
        """
        SELECT r.*, i.name as image_name, i.path as image_path
        FROM runs r
        JOIN images i ON r.image_id = i.id
        WHERE EXISTS (
            SELECT 1 FROM fft_results f
            WHERE f.run_id = r.id
            AND f.layer_id = ?
            AND f.filter_id = ?
        )
        ORDER BY r.timestamp
        """,
        (layer_id, filter_id)
    )

    runs = cursor.fetchall()
    conn.close()

    return [dict(run) for run in runs]

def get_fft_data_for_run(run_id, layer_id, filter_id):
    """
    Get FFT data for a specific run, layer, and filter.

    Args:
        run_id (int): The run ID
        layer_id (int): The layer ID
        filter_id (int): The filter ID

    Returns:
        tuple: (fft_data, freqs, dominant_frequencies)
    """
    conn = db.get_connection()
    cursor = conn.cursor()

    # Get FFT data
    cursor.execute(
        "SELECT fft_data FROM fft_results WHERE run_id = ? AND layer_id = ? AND filter_id = ?",
        (run_id, layer_id, filter_id)
    )
    fft_result = cursor.fetchone()

    # Get dominant frequencies
    cursor.execute(
        "SELECT * FROM dominant_frequencies WHERE run_id = ? AND layer_id = ? AND filter_id = ?",
        (run_id, layer_id, filter_id)
    )
    freq_data = cursor.fetchone()

    # Get run information for FPS
    cursor.execute("SELECT fps FROM runs WHERE id = ?", (run_id,))
    run_info = cursor.fetchone()

    conn.close()

    if not fft_result or not run_info:
        return None, None, None

    # Deserialize the FFT data
    fft_data = np.frombuffer(fft_result['fft_data'])

    # Generate frequency bins
    fps = run_info['fps']
    fft_length = len(fft_data)
    freqs = np.fft.fftfreq(fft_length, d=1/fps)
    freqs = np.fft.fftshift(freqs)  # Shift the zero frequency to the center

    return fft_data, freqs, freq_data

def plot_spectrum(run, layer_id, filter_id, output_dir=None, max_freq=35, auto_zoom=True):
    """
    Plot the spectrum for a specific run, layer, and filter.

    Args:
        run (dict): Run information
        layer_id (int): The layer ID
        filter_id (int): The filter ID
        output_dir (str, optional): Directory to save the plot
        max_freq (float, optional): Maximum frequency to display in Hz
        auto_zoom (bool, optional): Automatically zoom to the largest magnitude

    Returns:
        tuple: (fft_data, freqs) - The FFT data and frequency bins
    """
    fft_data, freqs, freq_data = get_fft_data_for_run(run['id'], layer_id, filter_id)

    if fft_data is None or freqs is None:
        print(f"No FFT data found for run {run['id']}, layer {layer_id}, filter {filter_id}.")
        return None, None

    # Plot the FFT data
    plt.figure(figsize=(12, 6))

    # Get positive frequencies only
    positive_mask = freqs > 0
    positive_freqs = freqs[positive_mask]
    positive_fft = np.abs(fft_data[positive_mask])

    # Filter frequencies to only show up to max_freq Hz
    freq_mask = positive_freqs <= max_freq
    plot_freqs = positive_freqs[freq_mask]
    plot_fft = positive_fft[freq_mask]

    # Find the largest magnitude for y-axis scaling
    max_magnitude = np.max(plot_fft)

    # Find the frequency with the largest magnitude
    max_magnitude_freq = plot_freqs[np.argmax(plot_fft)]

    # Plot the spectrum
    plt.bar(plot_freqs, plot_fft, width=0.05)
    plt.title(f"FFT Spectrum - {run['image_name']} - Layer {layer_id}, Filter {filter_id}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # Add more ticks on x-axis
    x_ticks = np.arange(0, max_freq + 1, 1)
    plt.xticks(x_ticks)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Set y-axis limit to slightly above the maximum magnitude for better visualization
    if auto_zoom:
        plt.ylim(0, max_magnitude * 1.1)  # Add 10% padding above the max magnitude

        # Add annotation for the maximum magnitude frequency
        plt.annotate(f"Max: {max_magnitude:.2f} at {max_magnitude_freq:.2f} Hz",
                    xy=(max_magnitude_freq, max_magnitude),
                    xytext=(max_magnitude_freq + 1, max_magnitude * 0.9),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)

    # Add vertical lines for dominant frequencies
    if freq_data:
        for i, freq_key in enumerate(['peak1_freq', 'peak2_freq', 'peak3_freq']):
            if freq_data[freq_key] and not np.isnan(freq_data[freq_key]):
                plt.axvline(x=freq_data[freq_key], color=['r', 'g', 'b'][i],
                           linestyle='--', linewidth=1,
                           label=f"Peak {i+1}: {freq_data[freq_key]:.2f} Hz")

    # Add vertical lines for GIF frequencies
    if run['gif_frequency1']:
        plt.axvline(x=run['gif_frequency1'], color='orange', linestyle='-', linewidth=1,
                   label=f"GIF Freq 1: {run['gif_frequency1']} Hz")
    if run['gif_frequency2']:
        plt.axvline(x=run['gif_frequency2'], color='purple', linestyle='-', linewidth=1,
                   label=f"GIF Freq 2: {run['gif_frequency2']} Hz")

    plt.legend()

    # Always save the plot to a directory
    if output_dir is None:
        # Default to 'results/spectrums' directory if not specified
        output_dir = 'results/spectrums'

    # Create a subdirectory for this specific filter and layer
    filter_layer_dir = os.path.join(output_dir, f"filter_{filter_id}_layer_{layer_id}")
    os.makedirs(filter_layer_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.splitext(os.path.basename(run['image_path']))[0]
    plot_path = os.path.join(filter_layer_dir, f"{image_name}_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"Saved spectrum plot to {plot_path}")

    plt.close()

    return fft_data, freqs

def generate_spectrum_gif(runs, layer_id, filter_id, output_file=None, max_freq=35, auto_zoom=True, duration=0.5):
    """
    Generate an animated GIF of spectrum plots across multiple runs.

    Args:
        runs (list): List of run dictionaries
        layer_id (int): The layer ID
        filter_id (int): The filter ID
        output_file (str, optional): Path to the output GIF file
        max_freq (float, optional): Maximum frequency to display in Hz
        auto_zoom (bool, optional): Automatically zoom to the largest magnitude

        duration (float, optional): Duration of each frame in seconds
    """
    if not runs:
        print("No runs to generate GIF from.")
        return

    # Create a temporary directory to store the frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate a plot for each run and save it to the temporary directory
        frame_paths = []
        max_magnitude_overall = 0

        # First pass: find the maximum magnitude across all runs if auto_zoom is True
        if auto_zoom:
            for i, run in enumerate(runs):
                fft_data, freqs, _ = get_fft_data_for_run(run['id'], layer_id, filter_id)

                if fft_data is None or freqs is None:
                    continue

                # Get positive frequencies only
                positive_mask = freqs > 0
                positive_freqs = freqs[positive_mask]
                positive_fft = np.abs(fft_data[positive_mask])

                # Filter frequencies to only show up to max_freq Hz
                freq_mask = positive_freqs <= max_freq
                plot_fft = positive_fft[freq_mask]

                # Update the maximum magnitude
                max_magnitude = np.max(plot_fft)
                max_magnitude_overall = max(max_magnitude_overall, max_magnitude)

        # Second pass: generate the plots with consistent y-axis scaling
        for i, run in enumerate(runs):
            fft_data, freqs, freq_data = get_fft_data_for_run(run['id'], layer_id, filter_id)

            if fft_data is None or freqs is None:
                continue

            # Plot the FFT data
            plt.figure(figsize=(12, 6))

            # Get positive frequencies only
            positive_mask = freqs > 0
            positive_freqs = freqs[positive_mask]
            positive_fft = np.abs(fft_data[positive_mask])

            # Filter frequencies to only show up to max_freq Hz
            freq_mask = positive_freqs <= max_freq
            plot_freqs = positive_freqs[freq_mask]
            plot_fft = positive_fft[freq_mask]

            # Find the largest magnitude for this run
            max_magnitude = np.max(plot_fft)

            # Find the frequency with the largest magnitude
            max_magnitude_freq = plot_freqs[np.argmax(plot_fft)]

            # Plot the spectrum
            plt.bar(plot_freqs, plot_fft, width=0.05)
            plt.title(f"FFT Spectrum - {run['image_name']} - Layer {layer_id}, Filter {filter_id}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")

            # Add more ticks on x-axis
            x_ticks = np.arange(0, max_freq + 1, 1)
            plt.xticks(x_ticks)
            plt.grid(axis='x', linestyle='--', alpha=0.7)

            # Set y-axis limit to slightly above the maximum magnitude for better visualization
            if auto_zoom:
                plt.ylim(0, max_magnitude_overall * 1.1)  # Use the overall maximum magnitude

                # Add annotation for the maximum magnitude frequency
                plt.annotate(f"Max: {max_magnitude:.2f} at {max_magnitude_freq:.2f} Hz",
                            xy=(max_magnitude_freq, max_magnitude),
                            xytext=(max_magnitude_freq + 1, max_magnitude * 0.9),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=10)

            # Add vertical lines for dominant frequencies
            if freq_data:
                for j, freq_key in enumerate(['peak1_freq', 'peak2_freq', 'peak3_freq']):
                    if freq_data[freq_key] and not np.isnan(freq_data[freq_key]):
                        plt.axvline(x=freq_data[freq_key], color=['r', 'g', 'b'][j],
                                   linestyle='--', linewidth=1,
                                   label=f"Peak {j+1}: {freq_data[freq_key]:.2f} Hz")

            # Add vertical lines for GIF frequencies
            if run['gif_frequency1']:
                plt.axvline(x=run['gif_frequency1'], color='orange', linestyle='-', linewidth=1,
                           label=f"GIF Freq 1: {run['gif_frequency1']} Hz")
            if run['gif_frequency2']:
                plt.axvline(x=run['gif_frequency2'], color='purple', linestyle='-', linewidth=1,
                           label=f"GIF Freq 2: {run['gif_frequency2']} Hz")

            plt.legend()

            # Save the frame to the temporary directory
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            plt.savefig(frame_path)
            frame_paths.append(frame_path)
            plt.close()

        if not frame_paths:
            print("No frames generated for GIF.")
            return

        # Create the output directory if it doesn't exist
        if output_file is None:
            # Default to 'results/spectrums' directory if not specified
            output_dir = 'results/spectrums'
            # Create a subdirectory for this specific filter and layer
            filter_layer_dir = os.path.join(output_dir, f"filter_{filter_id}_layer_{layer_id}")
            os.makedirs(filter_layer_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(filter_layer_dir, f"spectrum_animation_{timestamp}.gif")

        # Create the GIF
        # Create the GIF using imageio
        images = []
        for frame_path in frame_paths:
            images.append(imageio.imread(frame_path))

        # Save the GIF
        imageio.mimsave(output_file, images, duration=duration, loop=0)

        print(f"Generated animated GIF at {output_file}")

def export_to_csv(runs, layer_id, filter_id, output_file):
    """
    Export spectrum data for all runs to a CSV file.

    Args:
        runs (list): List of run dictionaries
        layer_id (int): The layer ID
        filter_id (int): The filter ID
        output_file (str): Path to the output CSV file
    """
    data = []

    for run in runs:
        fft_data, freqs, _ = get_fft_data_for_run(run['id'], layer_id, filter_id)

        if fft_data is None or freqs is None:
            continue

        # Get positive frequencies only
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_fft = np.abs(fft_data[positive_mask])

        # Create a row for each frequency
        for freq, magnitude in zip(positive_freqs, positive_fft):
            data.append({
                'run_id': run['id'],
                'image_name': run['image_name'],
                'layer_id': layer_id,
                'filter_id': filter_id,
                'frequency': freq,
                'magnitude': magnitude,
                'gif_frequency1': run['gif_frequency1'],
                'gif_frequency2': run['gif_frequency2'],
                'timestamp': run['timestamp']
            })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Exported spectrum data to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate spectrum data from the FFT database for a specific filter ID and layer.')
    parser.add_argument('filter_id', type=int, help='Filter ID to generate spectrum for')
    parser.add_argument('layer_id', type=int, help='Layer ID to generate spectrum for')
    parser.add_argument('--run-id', type=int, help='Specific run ID to use (optional)')
    parser.add_argument('--output-dir', help='Directory to save spectrum plots')
    parser.add_argument('--csv', help='Export spectrum data to CSV file')
    parser.add_argument('--max-freq', type=float, default=35, help='Maximum frequency to display in Hz (default: 35)')
    parser.add_argument('--list-only', action='store_true', help='Only list runs with data for the specified filter and layer')
    parser.add_argument('--no-zoom', action='store_true', help='Disable automatic zooming to the largest magnitude')
    parser.add_argument('--gif', action='store_true', help='Generate an animated GIF of spectrum plots across all runs')
    parser.add_argument('--gif-duration', type=float, default=0.5, help='Duration of each frame in the GIF in seconds (default: 0.5)')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to include in the GIF')

    args = parser.parse_args()

    # Initialize the database
    db.init_db()

    # Get runs with data for the specified filter and layer
    if args.run_id:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT r.*, i.name as image_name, i.path as image_path FROM runs r JOIN images i ON r.image_id = i.id WHERE r.id = ?",
            (args.run_id,)
        )
        run = cursor.fetchone()
        conn.close()

        if not run:
            print(f"Run with ID {args.run_id} not found.")
            return

        runs = [dict(run)]
    else:
        runs = get_runs_with_filter_layer(args.filter_id, args.layer_id)

    if not runs:
        print(f"No runs found with data for filter ID {args.filter_id} and layer ID {args.layer_id}.")
        return

    print(f"Found {len(runs)} runs with data for filter ID {args.filter_id} and layer ID {args.layer_id}:")
    print("-" * 100)
    print(f"{'ID':<5} {'Image':<20} {'Color':<8} {'FPS':<6} {'Method':<10} {'Freq1':<8} {'Freq2':<8} {'Timestamp':<20}")
    print("-" * 100)

    for run in runs:
        print(f"{run['id']:<5} {run['image_name']:<20} {run['color_format']:<8} {run['fps']:<6.1f} "
              f"{run['reduction_method']:<10} {run['gif_frequency1'] or 'N/A':<8} {run['gif_frequency2'] or 'N/A':<8} "
              f"{run['timestamp']}")

    # If list-only flag is set, exit after listing runs
    if args.list_only:
        return

    # Generate spectrum plots for each run
    for run in runs:
        plot_spectrum(run, args.layer_id, args.filter_id, args.output_dir, args.max_freq, not args.no_zoom)

    # Generate animated GIF if requested
    if args.gif:
        # Limit the number of frames if specified
        if args.max_frames and len(runs) > args.max_frames:
            print(f"Limiting GIF to {args.max_frames} frames (out of {len(runs)} runs)")
            gif_runs = runs[:args.max_frames]
        else:
            gif_runs = runs

        generate_spectrum_gif(gif_runs, args.layer_id, args.filter_id,
                             output_file=None,  # Use default naming
                             max_freq=args.max_freq,
                             auto_zoom=not args.no_zoom,
                             duration=args.gif_duration)

    # Export to CSV if requested
    if args.csv:
        export_to_csv(runs, args.layer_id, args.filter_id, args.csv)

def plot_and_save_spectrums(fourier_transformed_activations, output_dir, fps, dominant_frequencies,
                      gif_frequency1, gif_frequency2, specific_filter_id=None, specific_layer_id=None,
                      non_intermod=False):
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
    from src.core.signal_processing import is_harmonic_frequency, HarmonicType

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If specific layer ID is provided, only process that layer
    layer_ids = [specific_layer_id] if specific_layer_id is not None else fourier_transformed_activations.keys()

    for layer_id in layer_ids:
        # Skip if the layer ID doesn't exist in the data
        if layer_id not in fourier_transformed_activations:
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

            # Create subdirectory for this filter and layer
            filter_layer_dir = os.path.join(output_dir, f"filter_{filter_id}_layer_{layer_id}")
            os.makedirs(filter_layer_dir, exist_ok=True)

            plot_path = os.path.join(filter_layer_dir, f'spectrum.png')
            plt.savefig(plot_path)
            plt.close()

if __name__ == "__main__":
    main()
