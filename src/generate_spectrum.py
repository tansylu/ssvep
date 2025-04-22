#!/usr/bin/env python3
"""
Script to generate spectrum data from the FFT database for a specific filter ID and layer.
This script allows you to visualize the frequency spectrum for a specific filter and layer
across all runs or for specific runs in the database.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import db
from datetime import datetime

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

    # Save the plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.splitext(os.path.basename(run['image_path']))[0]
        plot_path = os.path.join(output_dir, f"{image_name}_layer_{layer_id}_filter_{filter_id}_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Saved spectrum plot to {plot_path}")
    else:
        plt.show()

    plt.close()

    return fft_data, freqs

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

    # Export to CSV if requested
    if args.csv:
        export_to_csv(runs, args.layer_id, args.filter_id, args.csv)

if __name__ == "__main__":
    main()
