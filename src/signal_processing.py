import numpy as np
from numpy.fft import fft
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

def find_dominant_frequencies(fourier_transformed_activations, fps):
    """
    Args:
        fourier_transformed_activations: {layer_id: np.array(num_filters, fft_length)}
        fps: sampling rate in Hz (frames per second)
    Returns:
        {layer_id: {filter_id: dominant_frequency}}
    """
    dominant_frequencies = {}
    for layer_id, layer_fft in fourier_transformed_activations.items():
        num_filters, fft_length = layer_fft.shape
        
        # Get frequency bins (convert to Hz by multiplying by fps)
        freqs = np.fft.fftfreq(fft_length, d=1/fps)  # freq in Hz
        freqs = np.fft.fftshift(freqs)  # Shift the zero frequency to the center
        
        dominant_frequencies[layer_id] = {}
        for filter_id in range(num_filters):
            # Get magnitudes of FFT for each filter
            filter_fft = layer_fft[filter_id]
            # Get the frequency with the highest magnitude (skip the DC component)
            # Ensure the FFT is shifted to avoid the DC component causing issues
            max_id = np.argmax(np.abs(filter_fft[1:])) + 1
            # Store the actual frequency (Hz) corresponding to the peak of the FFT
            dominant_frequencies[layer_id][filter_id] = abs(freqs[max_id])
    return dominant_frequencies

def save_dominant_frequencies_to_csv(dominant_frequencies, output_csv_path, image_path, gif_frequency):
    file_exists = os.path.exists(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write an empty row if desired
        writer.writerow([])
        # Then write the header row for the table data
        if not file_exists:
            writer.writerow(["Image", "Layer ID", "Filter ID", "Dominant Frequency", "GIF Frequency", "Difference", "Flag"])
        for layer_id in sorted(dominant_frequencies.keys()):
            filters = dominant_frequencies[layer_id]
            for filter_id in sorted(filters.keys()):
                dominant_frequency = filters[filter_id]
                difference = abs(dominant_frequency - gif_frequency)
                # Check if the dominant frequency is a harmonic of the GIF frequency
                is_harmonic = any(abs(dominant_frequency - n * gif_frequency) < 0.1 for n in range(1, 11))  # Adjust range as needed
                flag = "Different" if not is_harmonic else "Same"
                writer.writerow([image_path, layer_id, filter_id, f"{dominant_frequency:.2f}", gif_frequency, f"{difference:.2f}", flag])
        print(f"Dominant frequencies saved to '{output_csv_path}'")
