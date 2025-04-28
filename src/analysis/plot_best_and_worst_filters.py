import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import argparse
import sys

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import necessary modules
from src.database.db_stats import export_filter_stats_to_csv
from src.database import db
from src.analysis.frequency_similarity import calculate_frequency_similarity_score, get_similarity_category

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

def get_all_runs():
    """
    Get all runs from the database.

    Returns:
        list: List of run dictionaries
    """
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT r.*, i.path as image_path, i.name as image_name FROM runs r "
        "JOIN images i ON r.image_id = i.id "
        "ORDER BY r.timestamp"
    )
    runs = cursor.fetchall()
    conn.close()

    return [dict(run) for run in runs]

def get_fft_data_from_db(layer_id, filter_id, run_id=None):
    """
    Retrieve FFT data from the database for a specific layer and filter.
    If run_id is provided, only retrieve data for that run.
    Otherwise, retrieve data for all runs.

    Args:
        layer_id: Layer ID
        filter_id: Filter ID
        run_id: Run ID (optional)

    Returns:
        list: List of tuples (run, fft_data, freqs, fps, dominant_frequencies, similarity_score)
    """
    conn = db.get_connection()
    cursor = conn.cursor()

    # Get run information
    if run_id is not None:
        cursor.execute(
            "SELECT r.*, i.path as image_path, i.name as image_name FROM runs r "
            "JOIN images i ON r.image_id = i.id "
            "WHERE r.id = ?",
            (run_id,)
        )
        runs = cursor.fetchall()
    else:
        cursor.execute(
            "SELECT r.*, i.path as image_path, i.name as image_name FROM runs r "
            "JOIN images i ON r.image_id = i.id "
            "ORDER BY r.timestamp"
        )
        runs = cursor.fetchall()

    results = []

    for run in runs:
        run_dict = dict(run)
        run_id = run_dict['id']

        # Get FFT data
        cursor.execute(
            "SELECT fft_data FROM fft_results "
            "WHERE run_id = ? AND layer_id = ? AND filter_id = ?",
            (run_id, layer_id, filter_id)
        )
        result = cursor.fetchone()

        if not result:
            print(f"No FFT data found for run {run_id}, layer {layer_id}, filter {filter_id}.")
            continue

        # Get dominant frequencies
        cursor.execute(
            "SELECT peak1_freq, peak2_freq, peak3_freq, similarity_score FROM dominant_frequencies "
            "WHERE run_id = ? AND layer_id = ? AND filter_id = ?",
            (run_id, layer_id, filter_id)
        )
        freq_data = cursor.fetchone()

        # Deserialize the FFT data
        fft_data = np.frombuffer(result['fft_data'])

        # Generate frequency bins
        fps = run_dict['fps']
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

        results.append((run_dict, fft_data, freqs, fps, dominant_frequencies, similarity_score))

    conn.close()
    return results

def plot_filter_spectrum(layer_id, filter_id, output_dir, similarity_score=None, rank=None, is_best=True, run_id=None):
    """
    Plot the FFT spectrum for a specific filter using the actual FFT data from the database.
    If run_id is provided, only plot data for that run.
    Otherwise, plot data for all runs in a single figure.

    Creates both a static matplotlib plot and an interactive Plotly HTML file.

    Args:
        layer_id: Layer ID
        filter_id: Filter ID
        output_dir: Output directory for plots
        similarity_score: Similarity score (optional)
        rank: Rank in the list (optional)
        is_best: Whether this is one of the best filters (True) or worst filters (False)
        run_id: Run ID (optional)

    Returns:
        str: Path to the saved plot, or None if data not found
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get FFT data from the database
    results = get_fft_data_from_db(layer_id, filter_id, run_id)

    if not results:
        return None

    # Track the maximum magnitude for y-axis scaling
    max_magnitude = 0

    # Use different colors for each run
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Add rank and similarity score to the title if provided
    title = f'Layer {layer_id} Filter {filter_id} Spectrum - All Runs'
    if rank is not None and similarity_score is not None:
        rank_type = "Best" if is_best else "Worst"
        title = f'Layer {layer_id} Filter {filter_id} Spectrum ({rank_type} #{rank+1}, Score: {similarity_score:.4f})'

    # Create a Plotly figure for interactive visualization
    fig = go.Figure()

    # Create a matplotlib figure for static visualization
    plt.figure(figsize=(12, 6))

    # Plot data for each run
    for i, (run, fft_data, freqs, _, dominant_frequencies, run_similarity_score) in enumerate(results):
        # Get GIF frequencies
        gif_frequency1 = run['gif_frequency1']
        gif_frequency2 = run['gif_frequency2']

        # Recalculate similarity score if not provided
        if run_similarity_score is None and gif_frequency1 is not None and gif_frequency2 is not None:
            run_similarity_score = calculate_similarity_score(
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

        # Update max magnitude
        max_magnitude = max(max_magnitude, np.max(plot_fft) if len(plot_fft) > 0 else 0)

        # Get color for this run
        color = colors[i % len(colors)]
        run_name = f'Run {run["id"]} ({run["image_name"]})'

        # Add trace to Plotly figure
        fig.add_trace(go.Scatter(
            x=plot_freqs,
            y=plot_fft,
            mode='lines',
            name=run_name,
            line=dict(color=color, width=2),
            hovertemplate='Frequency: %{x:.2f} Hz<br>Magnitude: %{y:.2f}<extra></extra>'
        ))

        # Plot the FFT spectrum with a different color for each run (matplotlib)
        plt.plot(plot_freqs, plot_fft, label=run_name,
                 color=color, alpha=0.7, linewidth=1.5)

        # Add vertical lines for the dominant frequencies
        if dominant_frequencies:
            for j, freq in enumerate(dominant_frequencies):
                if freq <= 35:  # Only show frequencies up to 35 Hz
                    # Add to matplotlib
                    plt.axvline(x=freq, color=color, linestyle='--', linewidth=0.8,
                               alpha=0.5, label=f'Run {run["id"]} Peak {j+1}' if j == 0 else "")

                    # Add to Plotly
                    fig.add_trace(go.Scatter(
                        x=[freq, freq],
                        y=[0, max_magnitude * 1.1],
                        mode='lines',
                        line=dict(color=color, width=1, dash='dash'),
                        name=f'Run {run["id"]} Peak {j+1}' if j == 0 else f'Peak {j+1}',
                        showlegend=(j == 0)
                    ))

        # Generate harmonic ticks for visualization (only for the first run to avoid clutter)
        if i == 0 and gif_frequency1 is not None and gif_frequency2 is not None:
            max_harmonic = 12  # Show up to 11th harmonic
            harmonic_ticks1 = [n * gif_frequency1 for n in range(1, max_harmonic)]  # Start from 1 to skip zero
            harmonic_ticks2 = [n * gif_frequency2 for n in range(1, max_harmonic)]  # Start from 1 to skip zero

            for tick in harmonic_ticks1:
                if tick <= 35:  # Only show frequencies up to 35 Hz
                    # Add to matplotlib
                    plt.axvline(x=tick, color='r', linestyle=':', linewidth=0.5,
                               label='f1 harmonic' if tick == gif_frequency1 else "")

                    # Add to Plotly
                    fig.add_trace(go.Scatter(
                        x=[tick, tick],
                        y=[0, max_magnitude * 1.1],
                        mode='lines',
                        line=dict(color='red', width=1, dash='dot'),
                        name='f1 harmonic' if tick == gif_frequency1 else "",
                        showlegend=(tick == gif_frequency1)
                    ))

            for tick in harmonic_ticks2:
                if tick <= 35:  # Only show frequencies up to 35 Hz
                    # Add to matplotlib
                    plt.axvline(x=tick, color='g', linestyle=':', linewidth=0.5,
                               label='f2 harmonic' if tick == gif_frequency2 else "")

                    # Add to Plotly
                    fig.add_trace(go.Scatter(
                        x=[tick, tick],
                        y=[0, max_magnitude * 1.1],
                        mode='lines',
                        line=dict(color='green', width=1, dash='dot'),
                        name='f2 harmonic' if tick == gif_frequency2 else "",
                        showlegend=(tick == gif_frequency2)
                    ))

    # Configure Plotly layout
    fig.update_layout(
        title=title,
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        xaxis=dict(
            range=[0, 16],  # Set x-axis limit to 16 Hz
            dtick=1,  # Add ticks every 1 Hz
            gridcolor='lightgrey',
            gridwidth=1,
            griddash='dash'
        ),
        yaxis=dict(
            range=[0, max_magnitude * 1.1]  # Set y-axis limit with a small margin
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(l=50, r=50, t=80, b=100),
        hovermode='closest',
        template='plotly_white'
    )

    # Configure matplotlib plot
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # Add more ticks on x-axis
    # Create ticks every 1 Hz up to 35 Hz
    x_ticks = np.arange(0, 36, 1)
    plt.xticks(x_ticks)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Set x-axis limit to 16 Hz
    plt.xlim(0, 16)

    # Set y-axis limit with a small margin
    plt.ylim(0, max_magnitude * 1.1)

    # Add legend with smaller font size and outside the plot
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, frameon=True, fancybox=True, shadow=True)

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    # Save the plots
    prefix = "best" if is_best else "worst"
    if rank is not None:
        base_path = os.path.join(output_dir, f'{prefix}_{rank+1}_layer_{layer_id}_filter_{filter_id}_spectrum')
    else:
        base_path = os.path.join(output_dir, f'{prefix}_layer_{layer_id}_filter_{filter_id}_spectrum')

    # Save static plot
    static_path = f"{base_path}.png"
    plt.savefig(static_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save interactive plot
    interactive_path = f"{base_path}.html"
    fig.write_html(
        interactive_path,
        include_plotlyjs='cdn',  # Use CDN for plotly.js to reduce file size
        full_html=True,
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{prefix}_layer_{layer_id}_filter_{filter_id}_spectrum',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
    )

    return static_path

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
        run_id: Run ID to use for plotting (if None, use all runs)
    """
    # Load the data
    if not os.path.exists(stats_file):
        export_filter_stats_to_csv("filter_stats.csv")
        print(f"Filter statistics exported to {stats_file}")
        stats_file = "filter_stats.csv"

    stats_df = load_filter_stats(stats_file)

    # Sort by similarity score (ascending for worst, descending for best)
    worst_filters = stats_df.sort_values('Avg Similarity Score', ascending=True).head(num_filters)
    best_filters = stats_df.sort_values('Avg Similarity Score', ascending=False).head(num_filters)

    # Get all runs or filter by run_id
    if run_id is None:
        print(f"\nPlotting FFT responses for the {num_filters} best and worst performing filters from all runs:")
    else:
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
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Rank':<5}{'Layer':<10}{'Filter':<10}{'Similarity Score':<20}{'Plot Status':<15}{'Interactive Plot'}\n")
        f.write("-" * 100 + "\n")

        for i, (_, row) in enumerate(worst_filters.iterrows()):
            layer_id = int(row['Layer'])
            filter_id = int(row['Filter'])
            similarity_score = row['Avg Similarity Score']

            # Plot the filter
            plot_path = plot_filter_spectrum(
                layer_id, filter_id, worst_dir,
                similarity_score=similarity_score, rank=i, is_best=False, run_id=run_id
            )

            # Add to summary
            status = "Success" if plot_path else "Failed - No data found"

            # Generate interactive plot path
            if plot_path:
                interactive_path = plot_path.replace('.png', '.html')
                interactive_file = os.path.basename(interactive_path)
                interactive_info = f"<a href='{interactive_file}'>Interactive Plot</a>"
            else:
                interactive_info = "N/A"

            f.write(f"{i+1:<5}{layer_id:<10}{filter_id:<10}{similarity_score:<20.4f}{status:<15}{interactive_info}\n")

            # Print progress
            print(f"{i+1}. Layer {layer_id}, Filter {filter_id}: {similarity_score:.4f} - {status}")

    # Plot best filters
    print("\nPlotting best filters:")
    with open(best_summary_file, 'w') as f:
        f.write(f"Summary of {num_filters} best-performing filters\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Rank':<5}{'Layer':<10}{'Filter':<10}{'Similarity Score':<20}{'Plot Status':<15}{'Interactive Plot'}\n")
        f.write("-" * 100 + "\n")

        for i, (_, row) in enumerate(best_filters.iterrows()):
            layer_id = int(row['Layer'])
            filter_id = int(row['Filter'])
            similarity_score = row['Avg Similarity Score']

            # Plot the filter
            plot_path = plot_filter_spectrum(
                layer_id, filter_id, best_dir,
                similarity_score=similarity_score, rank=i, is_best=True, run_id=run_id
            )

            # Add to summary
            status = "Success" if plot_path else "Failed - No data found"

            # Generate interactive plot path
            if plot_path:
                interactive_path = plot_path.replace('.png', '.html')
                interactive_file = os.path.basename(interactive_path)
                interactive_info = f"<a href='{interactive_file}'>Interactive Plot</a>"
            else:
                interactive_info = "N/A"

            f.write(f"{i+1:<5}{layer_id:<10}{filter_id:<10}{similarity_score:<20.4f}{status:<15}{interactive_info}\n")

            # Print progress
            print(f"{i+1}. Layer {layer_id}, Filter {filter_id}: {similarity_score:.4f} - {status}")

    # Create an index.html file for each directory
    worst_index_file = os.path.join(worst_dir, "index.html")
    with open(worst_index_file, 'w') as f:
        f.write(f"<html><head><title>Worst {num_filters} Filters</title></head><body>\n")
        f.write(f"<h1>Worst {num_filters} Filters</h1>\n")
        f.write("<table border='1'>\n")
        f.write("<tr><th>Rank</th><th>Layer</th><th>Filter</th><th>Similarity Score</th><th>Static Plot</th><th>Interactive Plot</th></tr>\n")

        for i, (_, row) in enumerate(worst_filters.iterrows()):
            layer_id = int(row['Layer'])
            filter_id = int(row['Filter'])
            similarity_score = row['Avg Similarity Score']

            static_file = f"worst_{i+1}_layer_{layer_id}_filter_{filter_id}_spectrum.png"
            interactive_file = f"worst_{i+1}_layer_{layer_id}_filter_{filter_id}_spectrum.html"

            if os.path.exists(os.path.join(worst_dir, static_file)):
                f.write(f"<tr><td>{i+1}</td><td>{layer_id}</td><td>{filter_id}</td><td>{similarity_score:.4f}</td>")
                f.write(f"<td><a href='{static_file}'>Static Plot</a></td>")
                f.write(f"<td><a href='{interactive_file}'>Interactive Plot</a></td></tr>\n")

        f.write("</table></body></html>")

    best_index_file = os.path.join(best_dir, "index.html")
    with open(best_index_file, 'w') as f:
        f.write(f"<html><head><title>Best {num_filters} Filters</title></head><body>\n")
        f.write(f"<h1>Best {num_filters} Filters</h1>\n")
        f.write("<table border='1'>\n")
        f.write("<tr><th>Rank</th><th>Layer</th><th>Filter</th><th>Similarity Score</th><th>Static Plot</th><th>Interactive Plot</th></tr>\n")

        for i, (_, row) in enumerate(best_filters.iterrows()):
            layer_id = int(row['Layer'])
            filter_id = int(row['Filter'])
            similarity_score = row['Avg Similarity Score']

            static_file = f"best_{i+1}_layer_{layer_id}_filter_{filter_id}_spectrum.png"
            interactive_file = f"best_{i+1}_layer_{layer_id}_filter_{filter_id}_spectrum.html"

            if os.path.exists(os.path.join(best_dir, static_file)):
                f.write(f"<tr><td>{i+1}</td><td>{layer_id}</td><td>{filter_id}</td><td>{similarity_score:.4f}</td>")
                f.write(f"<td><a href='{static_file}'>Static Plot</a></td>")
                f.write(f"<td><a href='{interactive_file}'>Interactive Plot</a></td></tr>\n")

        f.write("</table></body></html>")

    # Create a main index.html file
    main_index_file = os.path.join(output_dir, "index.html")
    with open(main_index_file, 'w') as f:
        f.write("<html><head><title>Filter Spectrum Analysis</title></head><body>\n")
        f.write("<h1>Filter Spectrum Analysis</h1>\n")
        f.write("<p>This page provides links to the best and worst performing filters based on similarity score.</p>\n")
        f.write(f"<p><a href='best_filters/index.html'>Best {num_filters} Filters</a></p>\n")
        f.write(f"<p><a href='worst_filters/index.html'>Worst {num_filters} Filters</a></p>\n")
        f.write("</body></html>")

    print(f"\nBest filter plots saved to {best_dir}/")
    print(f"Best filter summary saved to {best_summary_file}")
    print(f"\nWorst filter plots saved to {worst_dir}/")
    print(f"Worst filter summary saved to {worst_summary_file}")
    print(f"\nHTML index files created for easy navigation:")
    print(f"- Main index: {main_index_file}")
    print(f"- Best filters index: {best_index_file}")
    print(f"- Worst filters index: {worst_index_file}")
    print(f"\nOpen {main_index_file} in a web browser to view the interactive plots.")

    return best_filters, worst_filters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot FFT responses of best and worst performing filters from database')
    parser.add_argument('--stats', type=str, default='filter_stats.csv',
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
