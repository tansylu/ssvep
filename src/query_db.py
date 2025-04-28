"""
Utility script to query the FFT results database.
This script provides command-line functionality to query and export data from the database.
"""

import argparse
import os
import db
import db_stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def list_images():
    """List all images in the database"""
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, path, name, created_at FROM images ORDER BY created_at DESC")
    images = cursor.fetchall()

    conn.close()

    if not images:
        print("No images found in the database.")
        return

    print(f"Found {len(images)} images in the database:")
    print("-" * 80)
    print(f"{'ID':<5} {'Name':<30} {'Created At':<20} {'Path'}")
    print("-" * 80)

    for img in images:
        print(f"{img['id']:<5} {img['name']:<30} {img['created_at']:<20} {img['path']}")

def list_runs(image_id=None):
    """List all runs in the database, optionally filtered by image ID"""
    conn = db.get_connection()
    cursor = conn.cursor()

    if image_id:
        cursor.execute(
            "SELECT r.id, r.timestamp, r.color_format, r.fps, r.reduction_method, "
            "r.gif_frequency1, r.gif_frequency2, r.created_at, i.name as image_name "
            "FROM runs r JOIN images i ON r.image_id = i.id "
            "WHERE r.image_id = ? "
            "ORDER BY r.created_at DESC",
            (image_id,)
        )
    else:
        cursor.execute(
            "SELECT r.id, r.timestamp, r.color_format, r.fps, r.reduction_method, "
            "r.gif_frequency1, r.gif_frequency2, r.created_at, i.name as image_name "
            "FROM runs r JOIN images i ON r.image_id = i.id "
            "ORDER BY r.created_at DESC"
        )

    runs = cursor.fetchall()

    conn.close()

    if not runs:
        print("No runs found in the database.")
        return

    print(f"Found {len(runs)} runs in the database:")
    print("-" * 100)
    print(f"{'ID':<5} {'Image':<20} {'Color':<8} {'FPS':<6} {'Method':<10} {'Freq1':<8} {'Freq2':<8} {'Timestamp':<20}")
    print("-" * 100)

    for run in runs:
        print(f"{run['id']:<5} {run['image_name']:<20} {run['color_format']:<8} {run['fps']:<6.1f} "
              f"{run['reduction_method']:<10} {run['gif_frequency1'] or 'N/A':<8} {run['gif_frequency2'] or 'N/A':<8} "
              f"{run['timestamp']}")

def export_run_to_csv(run_id, output_path=None):
    """Export a run to a CSV file"""
    if not output_path:
        output_path = f"run_{run_id}_export.csv"

    # Get run information
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT r.*, i.name as image_name, i.path as image_path "
        "FROM runs r JOIN images i ON r.image_id = i.id "
        "WHERE r.id = ?",
        (run_id,)
    )
    run = cursor.fetchone()

    if not run:
        print(f"Run with ID {run_id} not found.")
        conn.close()
        return

    # Get dominant frequencies
    cursor.execute(
        "SELECT * FROM dominant_frequencies "
        "WHERE run_id = ? "
        "ORDER BY layer_id, filter_id",
        (run_id,)
    )
    frequencies = cursor.fetchall()

    conn.close()

    if not frequencies:
        print(f"No dominant frequencies found for run ID {run_id}.")
        return

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([dict(f) for f in frequencies])

    # Add run information
    df['image_name'] = run['image_name']
    df['image_path'] = run['image_path']
    df['color_format'] = run['color_format']
    df['fps'] = run['fps']
    df['reduction_method'] = run['reduction_method']
    df['gif_frequency1'] = run['gif_frequency1']
    df['gif_frequency2'] = run['gif_frequency2']

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Exported run {run_id} to {output_path}")

def plot_fft_data(run_id, layer_id, filter_id, output_path=None):
    """Plot FFT data for a specific run, layer, and filter"""
    # Get run information
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT r.*, i.name as image_name "
        "FROM runs r JOIN images i ON r.image_id = i.id "
        "WHERE r.id = ?",
        (run_id,)
    )
    run = cursor.fetchone()

    if not run:
        print(f"Run with ID {run_id} not found.")
        conn.close()
        return

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
        return

    # Get dominant frequencies
    cursor.execute(
        "SELECT * FROM dominant_frequencies "
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
    freqs = np.fft.fftshift(freqs)  # Shift the zero frequency to the center

    # Plot the FFT data
    plt.figure(figsize=(12, 6))
    plt.bar(freqs[1:], np.abs(fft_data[1:]), width=0.05)  # Skip DC component
    plt.title(f"FFT Spectrum - {run['image_name']} - Layer {layer_id}, Filter {filter_id}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

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
    plt.grid(True, linestyle='--', alpha=0.7)

    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

def list_filter_stats(layer_id=None, filter_id=None, output_path=None, sort_by="diff_percent", limit=None):
    """List filter statistics from the database"""
    # Initialize the filter stats database if it doesn't exist
    db_stats.init_filter_stats_db()

    # Get filter statistics
    stats = db_stats.get_filter_stats()

    if not stats:
        print("No filter statistics found in the database.")
        return

    # Filter by layer_id if provided
    if layer_id is not None:
        stats = [s for s in stats if s['layer_id'] == layer_id]

    # Filter by filter_id if provided
    if filter_id is not None:
        stats = [s for s in stats if s['filter_id'] == filter_id]

    if not stats:
        print(f"No filter statistics found for the specified criteria.")
        return

    # Sort the results
    if sort_by == "diff_percent":
        stats.sort(key=lambda x: x['diff_percent'], reverse=True)
    elif sort_by == "layer_id":
        stats.sort(key=lambda x: x['layer_id'])
    elif sort_by == "filter_id":
        stats.sort(key=lambda x: x['filter_id'])
    elif sort_by == "total":
        stats.sort(key=lambda x: x['total'], reverse=True)

    # Apply limit if provided
    if limit is not None and limit > 0:
        stats = stats[:limit]

    # Export to CSV if output_path is provided
    if output_path:
        db_stats.export_filter_stats_to_csv(output_path)
        print(f"Filter statistics exported to {output_path}")
        return

    # Display the results
    print(f"Found {len(stats)} filter statistics:")
    print("-" * 80)
    print(f"{'Layer':<10} {'Filter':<10} {'Different':<10} {'Same':<10} {'Total':<10} {'Diff %':<10} {'Updated At':<20}")
    print("-" * 80)

    for stat in stats:
        print(f"{stat['layer_id']:<10} {stat['filter_id']:<10} {stat['different']:<10} {stat['same']:<10} "
              f"{stat['total']:<10} {stat['diff_percent']:<10.2f} {stat['updated_at']}")

def analyze_harmonics(run_id=None, image_id=None):
    """Analyze harmonic patterns in the database"""
    conn = db.get_connection()
    cursor = conn.cursor()

    query = """
    SELECT
        d.layer_id,
        COUNT(CASE WHEN d.is_harmonic = 1 THEN 1 END) as harmonic_count,
        COUNT(CASE WHEN d.is_harmonic = 0 THEN 1 END) as non_harmonic_count,
        COUNT(*) as total_count,
        ROUND(100.0 * COUNT(CASE WHEN d.is_harmonic = 1 THEN 1 END) / COUNT(*), 2) as harmonic_percentage
    FROM
        dominant_frequencies d
    """

    params = []
    if run_id:
        query += " WHERE d.run_id = ?"
        params.append(run_id)
    elif image_id:
        query += " JOIN runs r ON d.run_id = r.id WHERE r.image_id = ?"
        params.append(image_id)

    query += " GROUP BY d.layer_id ORDER BY d.layer_id"

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    if not results:
        print("No data found for analysis.")
        return

    print("Harmonic Analysis by Layer:")
    print("-" * 80)
    print(f"{'Layer':<10} {'Harmonic':<10} {'Non-Harmonic':<15} {'Total':<10} {'Harmonic %'}")
    print("-" * 80)

    for row in results:
        print(f"{row['layer_id']:<10} {row['harmonic_count']:<10} {row['non_harmonic_count']:<15} "
              f"{row['total_count']:<10} {row['harmonic_percentage']}%")

    # Plot the results
    layers = [row['layer_id'] for row in results]
    harmonic_pct = [row['harmonic_percentage'] for row in results]

    plt.figure(figsize=(10, 6))
    plt.bar(layers, harmonic_pct)
    plt.title("Harmonic Response by Layer")
    plt.xlabel("Layer ID")
    plt.ylabel("Harmonic Response (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(layers)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Query and analyze FFT results database")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List images command
    list_images_parser = subparsers.add_parser("list-images", help="List all images in the database")

    # List runs command
    list_runs_parser = subparsers.add_parser("list-runs", help="List all runs in the database")
    list_runs_parser.add_argument("--image-id", type=int, help="Filter runs by image ID")

    # List filter stats command
    filter_stats_parser = subparsers.add_parser("filter-stats", help="List filter statistics from the database")
    filter_stats_parser.add_argument("--layer-id", type=int, help="Filter by layer ID")
    filter_stats_parser.add_argument("--filter-id", type=int, help="Filter by filter ID")
    filter_stats_parser.add_argument("--output", help="Export statistics to CSV file")
    filter_stats_parser.add_argument("--sort-by", choices=["diff_percent", "layer_id", "filter_id", "total"],
                                   default="diff_percent", help="Sort results by this field")
    filter_stats_parser.add_argument("--limit", type=int, help="Limit the number of results")

    # Export run command
    export_parser = subparsers.add_parser("export", help="Export a run to CSV")
    export_parser.add_argument("run_id", type=int, help="Run ID to export")
    export_parser.add_argument("--output", help="Output CSV file path")

    # Plot FFT data command
    plot_parser = subparsers.add_parser("plot", help="Plot FFT data for a specific run, layer, and filter")
    plot_parser.add_argument("run_id", type=int, help="Run ID")
    plot_parser.add_argument("layer_id", type=int, help="Layer ID")
    plot_parser.add_argument("filter_id", type=int, help="Filter ID")
    plot_parser.add_argument("--output", help="Output image file path")

    # Analyze harmonics command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze harmonic patterns")
    analyze_parser.add_argument("--run-id", type=int, help="Analyze a specific run")
    analyze_parser.add_argument("--image-id", type=int, help="Analyze all runs for a specific image")

    # Initialize database command
    init_parser = subparsers.add_parser("init", help="Initialize the database")

    args = parser.parse_args()

    # Initialize the database if it doesn't exist
    db.init_db()

    if args.command == "list-images":
        list_images()
    elif args.command == "list-runs":
        list_runs(args.image_id)
    elif args.command == "filter-stats":
        list_filter_stats(args.layer_id, args.filter_id, args.output, args.sort_by, args.limit)
    elif args.command == "export":
        export_run_to_csv(args.run_id, args.output)
    elif args.command == "plot":
        plot_fft_data(args.run_id, args.layer_id, args.filter_id, args.output)
    elif args.command == "analyze":
        analyze_harmonics(args.run_id, args.image_id)
    elif args.command == "init":
        print("Database initialized successfully.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
