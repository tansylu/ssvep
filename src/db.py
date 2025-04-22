"""
Database module for storing and retrieving FFT results.
This module provides functions to initialize a SQLite database,
create the necessary tables, and store/retrieve FFT results.
"""

import os
import sqlite3
import numpy as np
import json
from datetime import datetime

# Database file path
DB_FILE = 'fft_results.db'

def get_connection():
    """
    Get a connection to the SQLite database.
    Creates the database file if it doesn't exist.

    Returns:
        sqlite3.Connection: A connection to the database
    """
    conn = sqlite3.connect(DB_FILE)
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    # Return rows as dictionaries
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initialize the database by creating the necessary tables if they don't exist.

    Tables:
    - images: Stores information about processed images
    - runs: Stores information about each processing run
    - fft_results: Stores the FFT results for each layer/filter
    - dominant_frequencies: Stores the dominant frequencies for each layer/filter
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Create images table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create runs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        color_format TEXT NOT NULL,
        fps REAL NOT NULL,
        gif_frequency1 REAL,
        gif_frequency2 REAL,
        reduction_method TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE
    )
    ''')

    # Create fft_results table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fft_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        layer_id INTEGER NOT NULL,
        filter_id INTEGER NOT NULL,
        fft_data BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES runs (id) ON DELETE CASCADE,
        UNIQUE (run_id, layer_id, filter_id)
    )
    ''')

    # Create dominant_frequencies table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dominant_frequencies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        layer_id INTEGER NOT NULL,
        filter_id INTEGER NOT NULL,
        peak1_freq REAL,
        peak2_freq REAL,
        peak3_freq REAL,
        is_harmonic BOOLEAN NOT NULL,
        harmonic_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES runs (id) ON DELETE CASCADE,
        UNIQUE (run_id, layer_id, filter_id)
    )
    ''')

    conn.commit()
    conn.close()

def get_or_create_image(image_path):
    """
    Get or create an image record in the database.

    Args:
        image_path (str): Path to the image file

    Returns:
        int: The image ID
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Extract the image name from the path
    image_name = os.path.basename(image_path)

    # Check if the image already exists
    cursor.execute("SELECT id FROM images WHERE path = ?", (image_path,))
    result = cursor.fetchone()

    if result:
        image_id = result['id']
    else:
        # Insert the new image
        cursor.execute(
            "INSERT INTO images (path, name) VALUES (?, ?)",
            (image_path, image_name)
        )
        image_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return image_id

def create_run(image_id, color_format, fps, reduction_method, gif_frequency1=None, gif_frequency2=None):
    """
    Create a new run record in the database.

    Args:
        image_id (int): The image ID
        color_format (str): The color format used (e.g., 'RGB', 'HSV')
        fps (float): Frames per second
        reduction_method (str): Method used to reduce spatial dimensions
        gif_frequency1 (float, optional): First GIF frequency
        gif_frequency2 (float, optional): Second GIF frequency

    Returns:
        int: The run ID
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Generate a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Insert the new run
    cursor.execute(
        "INSERT INTO runs (image_id, timestamp, color_format, fps, reduction_method, gif_frequency1, gif_frequency2) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (image_id, timestamp, color_format, fps, reduction_method, gif_frequency1, gif_frequency2)
    )
    run_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return run_id

def save_fft_results(run_id, fourier_transformed_activations):
    """
    Save FFT results to the database.

    Args:
        run_id (int): The run ID
        fourier_transformed_activations (dict): Dictionary of FFT results
            {layer_id: numpy_array(num_filters, fft_length)}
    """
    conn = get_connection()
    cursor = conn.cursor()

    for layer_id, layer_fft in fourier_transformed_activations.items():
        num_filters, _ = layer_fft.shape

        for filter_id in range(num_filters):
            # Get the FFT data for this filter
            fft_data = layer_fft[filter_id]

            # Serialize the numpy array to bytes
            serialized_data = fft_data.tobytes()

            # Insert or replace the FFT results
            cursor.execute(
                "INSERT OR REPLACE INTO fft_results (run_id, layer_id, filter_id, fft_data) "
                "VALUES (?, ?, ?, ?)",
                (run_id, layer_id, filter_id, serialized_data)
            )

    conn.commit()
    conn.close()

def save_dominant_frequencies(run_id, dominant_frequencies, gif_frequency1=None, gif_frequency2=None):
    """
    Save dominant frequencies to the database.

    Args:
        run_id (int): The run ID
        dominant_frequencies (dict): Dictionary of dominant frequencies
            {layer_id: {filter_id: [frequencies]}}
        gif_frequency1 (float, optional): First GIF frequency
        gif_frequency2 (float, optional): Second GIF frequency
    """
    from signal_processing import is_harmonic_frequency, HarmonicType

    conn = get_connection()
    cursor = conn.cursor()

    # Set harmonic detection parameters
    harmonic_tolerance = 1

    for layer_id in dominant_frequencies:
        filters = dominant_frequencies[layer_id]
        for filter_id in filters:
            peak_frequencies = filters[filter_id]  # List of top frequencies

            # Check if any peak is a harmonic
            is_harmonic = False
            harmonic_type = None

            if gif_frequency1 is not None and gif_frequency2 is not None:
                # Check for ANY harmonic
                is_harmonic_any = is_harmonic_frequency(
                    peak_frequencies=peak_frequencies,
                    freq1=gif_frequency1,
                    freq2=gif_frequency2,
                    harmonic_type=HarmonicType.ANY,
                    tolerance=harmonic_tolerance
                )

                # Check for FREQ1 harmonic
                is_harmonic_freq1 = is_harmonic_frequency(
                    peak_frequencies=peak_frequencies,
                    freq1=gif_frequency1,
                    freq2=gif_frequency2,
                    harmonic_type=HarmonicType.FREQ1,
                    tolerance=harmonic_tolerance
                )

                # Check for FREQ2 harmonic
                is_harmonic_freq2 = is_harmonic_frequency(
                    peak_frequencies=peak_frequencies,
                    freq1=gif_frequency1,
                    freq2=gif_frequency2,
                    harmonic_type=HarmonicType.FREQ2,
                    tolerance=harmonic_tolerance
                )

                # Check for INTERMOD harmonic
                is_harmonic_intermod = is_harmonic_frequency(
                    peak_frequencies=peak_frequencies,
                    freq1=gif_frequency1,
                    freq2=gif_frequency2,
                    harmonic_type=HarmonicType.INTERMOD,
                    tolerance=harmonic_tolerance
                )

                is_harmonic = is_harmonic_any

                # Determine the harmonic type
                if is_harmonic_freq1 and not is_harmonic_freq2 and not is_harmonic_intermod:
                    harmonic_type = "FREQ1"
                elif is_harmonic_freq2 and not is_harmonic_freq1 and not is_harmonic_intermod:
                    harmonic_type = "FREQ2"
                elif is_harmonic_intermod and not is_harmonic_freq1 and not is_harmonic_freq2:
                    harmonic_type = "INTERMOD"
                elif is_harmonic_any:
                    harmonic_type = "MULTIPLE"

            # Ensure we have exactly 3 peaks (pad with None if needed)
            while len(peak_frequencies) < 3:
                peak_frequencies.append(None)

            # Take only the first 3 peaks
            peak1, peak2, peak3 = peak_frequencies[:3]

            # Insert or replace the dominant frequencies
            cursor.execute(
                "INSERT OR REPLACE INTO dominant_frequencies "
                "(run_id, layer_id, filter_id, peak1_freq, peak2_freq, peak3_freq, is_harmonic, harmonic_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, layer_id, filter_id, peak1, peak2, peak3, is_harmonic, harmonic_type)
            )

    conn.commit()
    conn.close()

def get_fft_data(run_id, layer_id, filter_id):
    """
    Retrieve FFT data from the database.

    Args:
        run_id (int): The run ID
        layer_id (int): The layer ID
        filter_id (int): The filter ID

    Returns:
        numpy.ndarray: The FFT data as a numpy array
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT fft_data FROM fft_results WHERE run_id = ? AND layer_id = ? AND filter_id = ?",
        (run_id, layer_id, filter_id)
    )
    result = cursor.fetchone()

    conn.close()

    if result:
        # Deserialize the numpy array from bytes
        fft_data = np.frombuffer(result['fft_data'])
        return fft_data
    else:
        return None

def get_dominant_frequencies(run_id=None, layer_id=None, filter_id=None, is_harmonic=None):
    """
    Retrieve dominant frequencies from the database.

    Args:
        run_id (int, optional): Filter by run ID
        layer_id (int, optional): Filter by layer ID
        filter_id (int, optional): Filter by filter ID
        is_harmonic (bool, optional): Filter by harmonic status

    Returns:
        list: List of dominant frequencies matching the criteria
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM dominant_frequencies WHERE 1=1"
    params = []

    if run_id is not None:
        query += " AND run_id = ?"
        params.append(run_id)

    if layer_id is not None:
        query += " AND layer_id = ?"
        params.append(layer_id)

    if filter_id is not None:
        query += " AND filter_id = ?"
        params.append(filter_id)

    if is_harmonic is not None:
        query += " AND is_harmonic = ?"
        params.append(is_harmonic)

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    # Convert the results to a list of dictionaries
    return [dict(row) for row in results]

def get_runs_for_image(image_path):
    """
    Get all runs for a specific image.

    Args:
        image_path (str): Path to the image file

    Returns:
        list: List of runs for the image
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT r.* FROM runs r "
        "JOIN images i ON r.image_id = i.id "
        "WHERE i.path = ?",
        (image_path,)
    )
    results = cursor.fetchall()

    conn.close()

    # Convert the results to a list of dictionaries
    return [dict(row) for row in results]

def export_to_csv(run_id, output_path):
    """
    Export dominant frequencies for a specific run to a CSV file.

    Args:
        run_id (int): The run ID
        output_path (str): Path to the output CSV file
    """
    import csv

    conn = get_connection()
    cursor = conn.cursor()

    # Get run information
    cursor.execute(
        "SELECT r.*, i.path as image_path FROM runs r "
        "JOIN images i ON r.image_id = i.id "
        "WHERE r.id = ?",
        (run_id,)
    )
    run = cursor.fetchone()

    if not run:
        conn.close()
        raise ValueError(f"Run with ID {run_id} not found")

    # Get dominant frequencies for this run
    cursor.execute(
        "SELECT * FROM dominant_frequencies "
        "WHERE run_id = ? "
        "ORDER BY layer_id, filter_id",
        (run_id,)
    )
    frequencies = cursor.fetchall()

    conn.close()

    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow([
            "Image", "Layer ID", "Filter ID",
            "Peak 1 Freq", "Peak 2 Freq", "Peak 3 Freq",
            "GIF Frequency 1", "GIF Frequency 2", "Flag"
        ])

        # Write data
        for freq in frequencies:
            writer.writerow([
                run['image_path'],
                freq['layer_id'],
                freq['filter_id'],
                freq['peak1_freq'],
                freq['peak2_freq'],
                freq['peak3_freq'],
                run['gif_frequency1'],
                run['gif_frequency2'],
                "Same" if freq['is_harmonic'] else "Different"
            ])
