"""
Test script for the FFT results database.
This script tests the database functionality by creating a simple test run.
"""

import os
import numpy as np
import db
from signal_processing import save_fft_results_to_db

def create_test_data():
    """Create test data for the database"""
    # Create a test image path
    image_path = "test_image.jpg"
    
    # Create test FFT data
    fourier_transformed_activations = {
        0: np.random.rand(10, 24),  # 10 filters, 24 frames
        1: np.random.rand(20, 24)   # 20 filters, 24 frames
    }
    
    # Create test dominant frequencies
    dominant_frequencies = {}
    for layer_id in fourier_transformed_activations:
        num_filters = fourier_transformed_activations[layer_id].shape[0]
        dominant_frequencies[layer_id] = {}
        for filter_id in range(num_filters):
            # Generate random frequencies between 1 and 20 Hz
            dominant_frequencies[layer_id][filter_id] = [
                np.random.uniform(1, 20),
                np.random.uniform(1, 20),
                np.random.uniform(1, 20)
            ]
    
    return image_path, fourier_transformed_activations, dominant_frequencies

def test_database():
    """Test the database functionality"""
    print("Initializing database...")
    db.init_db()
    
    print("Creating test data...")
    image_path, fourier_transformed_activations, dominant_frequencies = create_test_data()
    
    print("Saving test data to database...")
    run_id = save_fft_results_to_db(
        image_path=image_path,
        fourier_transformed_activations=fourier_transformed_activations,
        dominant_frequencies=dominant_frequencies,
        color_format="RGB",
        fps=24,
        reduction_method="mean",
        gif_frequency1=5,
        gif_frequency2=6
    )
    
    print(f"Test data saved to database with run ID: {run_id}")
    
    # Verify that the data was saved correctly
    print("Verifying data in database...")
    
    # Check that the run exists
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
    run = cursor.fetchone()
    
    if run:
        print(f"Run {run_id} found in database.")
        print(f"  Color format: {run['color_format']}")
        print(f"  FPS: {run['fps']}")
        print(f"  Reduction method: {run['reduction_method']}")
        print(f"  GIF frequency 1: {run['gif_frequency1']}")
        print(f"  GIF frequency 2: {run['gif_frequency2']}")
    else:
        print(f"Run {run_id} not found in database!")
    
    # Check that FFT results exist
    cursor.execute("SELECT COUNT(*) as count FROM fft_results WHERE run_id = ?", (run_id,))
    fft_count = cursor.fetchone()['count']
    
    if fft_count > 0:
        print(f"Found {fft_count} FFT results for run {run_id}.")
    else:
        print(f"No FFT results found for run {run_id}!")
    
    # Check that dominant frequencies exist
    cursor.execute("SELECT COUNT(*) as count FROM dominant_frequencies WHERE run_id = ?", (run_id,))
    freq_count = cursor.fetchone()['count']
    
    if freq_count > 0:
        print(f"Found {freq_count} dominant frequency records for run {run_id}.")
    else:
        print(f"No dominant frequencies found for run {run_id}!")
    
    conn.close()
    
    print("Database test completed.")

if __name__ == "__main__":
    test_database()
