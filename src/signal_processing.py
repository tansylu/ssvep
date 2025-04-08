import numpy as np
from numpy.fft import fft
from scipy.signal import find_peaks
import os
import csv

def perform_fourier_transform(activations, reduction_method='mean'):
    """
    Performs FFT on activation time series for each layer and filter.
    Args:
        activations: {layer_id: [frame1_tensor(1,filters,height,width), frame2_tensor...]}
        reduction_method: Method to reduce spatial dimensions ('mean', 'sum', 'max', 'min', 'median')
    Returns:
        {layer_id: numpy_array(num_filters, fft_length)}
    """
    reduction_methods = {
        'mean': np.mean,
        'sum': np.sum,
        'max': np.max,
        'min': np.min,
        'median': np.median
    }
    
    if reduction_method not in reduction_methods:
        raise ValueError(f"Invalid reduction method: {reduction_method}. Choose from 'mean', 'sum', 'max', 'min', 'median'.")#try l2, better csv plots,
    
    reduce_fn = reduction_methods[reduction_method]
    
    fourier_transformed_activations = {}
    for layer_id, frames in activations.items():
        num_filters = frames[0].shape[1]
        num_frames = len(frames)
        
        # Initialize the Fourier transformed activations array
        fourier_transformed_activations[layer_id] = np.zeros((num_filters, num_frames))
        
        # Iterate over each filter
        for filter_id in range(num_filters):
            # Extract the temporal sequence for each filter using the specified reduction method
            temporal_sequence = [reduce_fn(frame[0, filter_id, :, :]) for frame in frames]
            
            # Perform Fourier Transform on the temporal sequence
            fourier_transformed_activations[layer_id][filter_id] = np.abs(fft(temporal_sequence))
    
    return fourier_transformed_activations

def find_dominant_frequencies(fourier_transformed_activations, fps, min_prominence=0.01):
    """
    Identifies dominant frequency peaks for each filter in each layer using peak detection.

    Args:
        fourier_transformed_activations (dict): {layer_id: np.array(num_filters, fft_length)}
        fps (float): Sampling rate in Hz (frames per second)
        min_prominence (float): Minimum prominence of a peak (relative to max magnitude)

    Returns:
        dict: {layer_id: {filter_id: dominant_frequency (Hz)}}
    """
    dominant_frequencies = {}

    for layer_id, layer_fft in fourier_transformed_activations.items():
        num_filters, fft_length = layer_fft.shape
        # Generate frequency bins
        freqs = np.fft.fftfreq(fft_length, d=1/fps)
        freqs = np.fft.fftshift(freqs)  # Center zero frequency

        dominant_frequencies[layer_id] = {}

        for filter_id in range(num_filters):
            fft_vals = layer_fft[filter_id]
            magnitudes = np.fft.fftshift(np.abs(fft_vals))

            # Remove DC component explicitly (center of fftshifted array)
            center = fft_length // 2
            magnitudes[center] = 0

            # Find peaks in the magnitude spectrum
            peak_indices, properties = find_peaks(
                magnitudes,
                prominence=min_prominence * np.max(magnitudes)
            )
            if len(peak_indices) == 0:
                dominant_freq = 0
            else:
                # Take the highest peak
                dominant_peak = peak_indices[np.argmax(magnitudes[peak_indices])]
                dominant_freq = abs(freqs[dominant_peak])

            dominant_frequencies[layer_id][filter_id] = dominant_freq
    return dominant_frequencies

def save_dominant_frequencies_to_csv(dominant_frequencies, output_csv_path, image_path, gif_frequency1,gif_frequency2):
    file_exists = os.path.exists(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write an empty row if desired
        writer.writerow([])
        # Then write the header row for the table data
        if not file_exists:
            writer.writerow(["Image", "Layer ID", "Filter ID", "Dominant Frequency", "GIF Frequency 1","GIF Frequency 2", "Difference", "Flag"])
        for layer_id in sorted(dominant_frequencies.keys()):
            filters = dominant_frequencies[layer_id]
            for filter_id in sorted(filters.keys()):
                dominant_frequency = filters[filter_id]
                difference = abs(dominant_frequency - gif_frequency1)
                # Check if the dominant frequency is a harmonic of the GIF frequency
                harmonic_tolerance = 0.1
                harmonics_freq1 = [n * gif_frequency1 for n in range(1, 11)]
                harmonics_freq2 = [n * gif_frequency2 for n in range(1, 11)]

                is_harmonic = any(abs(dominant_frequency - h) < harmonic_tolerance for h in harmonics_freq1 + harmonics_freq2)
                flag = "Different" if not is_harmonic else "Same"
                if flag == "Different":
                    # Save the dominant frequency and the difference
                    writer.writerow([image_path, layer_id, filter_id, f"{dominant_frequency:.2f}", gif_frequency1,gif_frequency2, f"{difference:.2f}", flag])
        # print(f"Dominant frequencies saved to '{output_csv_path}'")
            