import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Add the src directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary modules
import db
from frequency_similarity import calculate_frequency_similarity_score, get_similarity_category

def load_filter_stats(stats_file):
    """Load filter statistics from CSV file"""
    print(f"Loading filter statistics from {stats_file}")
    stats_df = pd.read_csv(stats_file)
    return stats_df

def calculate_similarity_score(fft_data, freqs, target_freq1, target_freq2, tolerance=0.5):
    """
    Calculate similarity score using the full spectrum.

    Args:
        fft_data: FFT data
        freqs: Frequency bins
        target_freq1: First target frequency
        target_freq2: Second target frequency
        tolerance: Frequency tolerance

    Returns:
        float: Similarity score
    """
    # Get magnitudes of the FFT data
    magnitudes = np.abs(fft_data)

    # Get only positive frequencies
    positive_mask = freqs > 0
    positive_freqs = freqs[positive_mask]
    positive_magnitudes = magnitudes[positive_mask]

    # Calculate similarity using the full spectrum
    similarity_score, _ = calculate_frequency_similarity_score(
        frequencies=positive_freqs,
        magnitudes=positive_magnitudes,
        target_freq1=target_freq1,
        target_freq2=target_freq2,
        tolerance=tolerance
    )

    return similarity_score

def get_fft_data_from_db(run_id, layer_id, filter_id):
    """
    Retrieve FFT data from the database for a specific run, layer, and filter.

    Args:
        run_id: Run ID
        layer_id: Layer ID
        filter_id: Filter ID

    Returns:
        tuple: (fft_data, freqs, fps, dominant_frequencies)
    """
    conn = db.get_connection()
    cursor = conn.cursor()

    # Get run information for FPS
    cursor.execute(
        "SELECT r.*, i.path as image_path FROM runs r "
        "JOIN images i ON r.image_id = i.id "
        "WHERE r.id = ?",
        (run_id,)
    )
    run = cursor.fetchone()

    if not run:
        print(f"Run with ID {run_id} not found.")
        conn.close()
        return None, None, None, None, None

    # Get FFT data
    cursor.execute(
        "SELECT fft_data FROM fft_results "
        "WHERE run_id = ? AND layer_id = ? AND filter_id = ?",
        (run_id, layer_id, filter_id)
    )
    result = cursor.fetchone()

    if not result:
        print(f"No FFT data found for run {run_id}, layer {layer_id}, filter {filter_id}.")
        conn.close()
        return None, None, None, None, None

    # Get dominant frequencies
    cursor.execute(
        "SELECT peak1_freq, peak2_freq, peak3_freq, similarity_score FROM dominant_frequencies "
        "WHERE run_id = ? AND layer_id = ? AND filter_id = ?",
        (run_id, layer_id, filter_id)
    )
    freq_data = cursor.fetchone()

    conn.close()

    # Deserialize the FFT data
    fft_data = np.frombuffer(result['fft_data'])

    # Generate frequency bins
    fps = run['fps']
    fft_length = len(fft_data)
    freqs = np.fft.fftfreq(fft_length, d=1/fps)

    # Extract dominant frequencies
    dominant_frequencies = []
    if freq_data:
        for i in range(3):
            freq = freq_data[f'peak{i+1}_freq']
            if freq is not None and not np.isnan(freq):
                dominant_frequencies.append(freq)

        # Get similarity score
        similarity_score = freq_data['similarity_score']
    else:
        similarity_score = None

    return fft_data, freqs, fps, dominant_frequencies, run

def plot_filter_spectrum(run_id, layer_id, filter_id, output_dir, similarity_score=None, rank=None, is_best=True):
    """
    Plot the FFT spectrum for a specific filter using the actual FFT data from the database.

    Args:
        run_id: Run ID
        layer_id: Layer ID
        filter_id: Filter ID
        output_dir: Output directory for plots
        similarity_score: Similarity score (optional)
        rank: Rank in the list (optional)
        is_best: Whether this is one of the best filters (True) or worst filters (False)

    Returns:
        str: Path to the saved plot, or None if data not found
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get FFT data from the database
    fft_data, freqs, fps, dominant_frequencies, run = get_fft_data_from_db(run_id, layer_id, filter_id)

    if fft_data is None:
        return None

    # Get GIF frequencies
    gif_frequency1 = run['gif_frequency1']
    gif_frequency2 = run['gif_frequency2']

    # Recalculate similarity score if not provided
    if similarity_score is None and gif_frequency1 is not None and gif_frequency2 is not None:
        similarity_score = calculate_similarity_score(
            fft_data, freqs, gif_frequency1, gif_frequency2
        )

    # Get only positive frequencies (excluding DC component)
    positive_mask = freqs > 0
    positive_freqs = freqs[positive_mask]
    positive_fft = np.abs(fft_data[positive_mask])

    # Filter frequencies to only show up to 35 Hz
    freq_mask = positive_freqs <= 35
    plot_freqs = positive_freqs[freq_mask]
    plot_fft = positive_fft[freq_mask]

    # Plot the FFT spectrum
    plt.figure(figsize=(10, 5))
    plt.bar(plot_freqs, plot_fft, width=0.05, label=f'Filter {filter_id}')

    # Add rank and similarity score to the title if provided
    title = f'Layer {layer_id} Filter {filter_id} Spectrum'
    if rank is not None and similarity_score is not None:
        rank_type = "Best" if is_best else "Worst"
        title += f' ({rank_type} #{rank+1}, Score: {similarity_score:.4f})'

    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    # Add more ticks on x-axis
    # Create ticks every 1 Hz up to 35 Hz
    x_ticks = np.arange(0, 36, 1)
    plt.xticks(x_ticks)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Set x-axis limit to 16 Hz
    plt.xlim(0, 16)

    # Generate harmonic ticks for visualization
    if gif_frequency1 is not None and gif_frequency2 is not None:
        max_harmonic = 12  # Show up to 11th harmonic
        harmonic_ticks1 = [n * gif_frequency1 for n in range(1, max_harmonic)]  # Start from 1 to skip zero
        harmonic_ticks2 = [n * gif_frequency2 for n in range(1, max_harmonic)]  # Start from 1 to skip zero

        for tick in harmonic_ticks1:
            plt.axvline(x=tick, color='r', linestyle='--', linewidth=0.5,
                       label='f1 harmonic' if tick == gif_frequency1 else "")

        for tick in harmonic_ticks2:
            plt.axvline(x=tick, color='g', linestyle='--', linewidth=0.5,
                       label='f2 harmonic' if tick == gif_frequency2 else "")

    # Save the plot
    prefix = "best" if is_best else "worst"
    if rank is not None:
        plot_path = os.path.join(output_dir, f'{prefix}_{rank+1}_layer_{layer_id}_filter_{filter_id}_spectrum.png')
    else:
        plot_path = os.path.join(output_dir, f'{prefix}_layer_{layer_id}_filter_{filter_id}_spectrum.png')

    plt.savefig(plot_path)
    plt.close()

    return plot_path

def get_latest_run_id():
    """Get the ID of the latest run in the database"""
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM runs ORDER BY timestamp DESC LIMIT 1")
    result = cursor.fetchone()

    conn.close()

    if result:
        return result['id']
    else:
        return None

def plot_best_and_worst_filters(stats_file, num_filters=20, output_dir='filter_comparison', run_id=None):
    """
    Plot FFT responses of both the best and worst performing filters using actual FFT data from the database.

    Args:
        stats_file: Path to the filter statistics CSV file
        num_filters: Number of best/worst filters to plot
        output_dir: Output directory for plots
        run_id: Run ID to use for plotting (if None, use the latest run)
    """
    # Load the data
    stats_df = load_filter_stats(stats_file)

    # Sort by similarity score (ascending for worst, descending for best)
    worst_filters = stats_df.sort_values('Avg Similarity Score', ascending=True).head(num_filters)
    best_filters = stats_df.sort_values('Avg Similarity Score', ascending=False).head(num_filters)

    # If run_id is not provided, use the latest run
    if run_id is None:
        run_id = get_latest_run_id()
        if run_id is None:
            print("No runs found in the database.")
            return

    print(f"\nPlotting FFT responses for the {num_filters} best and worst performing filters from run ID {run_id}:")

    # Create output directories
    best_dir = os.path.join(output_dir, 'best_filters')
    worst_dir = os.path.join(output_dir, 'worst_filters')
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    # Create summary files
    best_summary_file = os.path.join(best_dir, "best_filters_summary.txt")
    worst_summary_file = os.path.join(worst_dir, "worst_filters_summary.txt")

    # Plot worst filters
    print("\nPlotting worst filters:")
    with open(worst_summary_file, 'w') as f:
        f.write(f"Summary of {num_filters} worst-performing filters\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Rank':<5}{'Layer':<10}{'Filter':<10}{'Similarity Score':<20}{'Plot Status'}\n")
        f.write("-" * 80 + "\n")

        for i, (_, row) in enumerate(worst_filters.iterrows()):
            layer_id = int(row['Layer'])
            filter_id = int(row['Filter'])
            similarity_score = row['Avg Similarity Score']

            # Plot the filter
            plot_path = plot_filter_spectrum(
                run_id, layer_id, filter_id, worst_dir,
                similarity_score=similarity_score, rank=i, is_best=False
            )

            # Add to summary
            status = "Success" if plot_path else "Failed - No data found"
            f.write(f"{i+1:<5}{layer_id:<10}{filter_id:<10}{similarity_score:<20.4f}{status}\n")

            # Print progress
            print(f"{i+1}. Layer {layer_id}, Filter {filter_id}: {similarity_score:.4f} - {status}")

    # Plot best filters
    print("\nPlotting best filters:")
    with open(best_summary_file, 'w') as f:
        f.write(f"Summary of {num_filters} best-performing filters\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Rank':<5}{'Layer':<10}{'Filter':<10}{'Similarity Score':<20}{'Plot Status'}\n")
        f.write("-" * 80 + "\n")

        for i, (_, row) in enumerate(best_filters.iterrows()):
            layer_id = int(row['Layer'])
            filter_id = int(row['Filter'])
            similarity_score = row['Avg Similarity Score']

            # Plot the filter
            plot_path = plot_filter_spectrum(
                run_id, layer_id, filter_id, best_dir,
                similarity_score=similarity_score, rank=i, is_best=True
            )

            # Add to summary
            status = "Success" if plot_path else "Failed - No data found"
            f.write(f"{i+1:<5}{layer_id:<10}{filter_id:<10}{similarity_score:<20.4f}{status}\n")

            # Print progress
            print(f"{i+1}. Layer {layer_id}, Filter {filter_id}: {similarity_score:.4f} - {status}")

    print(f"\nBest filter plots saved to {best_dir}/")
    print(f"Best filter summary saved to {best_summary_file}")
    print(f"\nWorst filter plots saved to {worst_dir}/")
    print(f"Worst filter summary saved to {worst_summary_file}")

    return best_filters, worst_filters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot FFT responses of best and worst performing filters from database')
    parser.add_argument('--stats', type=str, default='filter_stats_20250427_164606.csv',
                        help='Path to the filter statistics CSV file')
    parser.add_argument('--num', type=int, default=20,
                        help='Number of best/worst filters to plot')
    parser.add_argument('--output', type=str, default='filter_comparison',
                        help='Output directory for plots')
    parser.add_argument('--run-id', type=int, default=None,
                        help='Run ID to use for plotting (if not provided, use the latest run)')

    args = parser.parse_args()

    # Plot the best and worst filters
    best_filters, worst_filters = plot_best_and_worst_filters(
        args.stats, args.num, args.output, args.run_id
    )
