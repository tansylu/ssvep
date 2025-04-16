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

def find_peaks_two_neighbours(signal, threshold_factor=2.0):
    """
    Custom peak detection that only looks at immediate neighbours.
    Args:
        signal: 1D numpy array containing the signal
        threshold_factor: A point is considered a peak if its value is greater than 
                         threshold_factor times the average of its neighbours
    Returns:
        numpy array: Indices of detected peaks
    """
    # Signal must be at least 3 points long to find peaks
    if len(signal) < 3:
        return np.array([])
    
    peaks = []    

    # Check each point (except first and last) against neighbours
    for i in range(1, len(signal) - 1):
        # Calculate average of neighbours
        neighbor_avg = (signal[i-1] + signal[i+1]) / 2
        
        # Check if current point is greater than both neighbours
        # and greater than threshold_factor times the average of neighbours
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold_factor * neighbor_avg:
            peaks.append(i)
    
    return np.array(peaks)

def find_peaks_four_neighbours(signal, threshold_factor=1.5):
    """
    Custom peak detection that looks at two neighbours on each side.
    Args:
        signal: 1D numpy array containing the signal
        threshold_factor: A point is considered a peak if its value is greater than threshold_factor times the average of its neighbours
    Returns:
        numpy array: Indices of detected peaks
    """
    # Signal must be at least 5 points long to check 2 neighbours on each side
    if len(signal) < 5:
        return np.array([])
    
    peaks = []
    
    # Check each point against two neighbours on each side
    for i in range(2, len(signal) - 2):
        # Calculate average of all four neighbours
        neighbor_avg = (signal[i-2] + signal[i-1] + signal[i+1] + signal[i+2]) / 4
        
        # Check if current point is greater than ALL four neighbours
        is_local_max = (signal[i] > signal[i-2] and 
                        signal[i] > signal[i-1] and 
                        signal[i] > signal[i+1] and 
                        signal[i] > signal[i+2])
        
        # And check if it's significantly higher than the average of its neighbours
        is_significant = signal[i] > threshold_factor * neighbor_avg
        
        if is_local_max and is_significant:
            peaks.append(i)
    
    return np.array(peaks)

def find_peaks_snr(signal, min_snr=3.0):
    """
    Detects peaks based on local signal-to-noise ratio.
    Args:
        signal: 1D numpy array containing the signal
        min_snr: Minimum SNR to consider a point as a peak
    Returns:
        numpy array: Indices of detected peaks
    """
    if len(signal) < 5:  # Need at least 5 points to estimate noise
        return np.array([])
    
    peaks = []
    for i in range(2, len(signal) - 2):
        # Calculate local noise level (using points not immediately adjacent)
        local_noise = np.std([signal[i-2], signal[i-1], signal[i+1], signal[i+2]])
        if local_noise == 0:  # Avoid division by zero
            local_noise = 1e-10
            
        # Calculate SNR
        snr = signal[i] / local_noise
        
        if (signal[i] > signal[i-1] and signal[i] > signal[i+1] and snr > min_snr):
            peaks.append(i)
    
    return np.array(peaks)


def find_dominant_frequencies(fourier_transformed_activations, fps, threshold_factor=2.0, num_peaks=3, min_snr=3.0, method='custom'):
    """
    Identifies dominant frequency peaks for each filter in each layer using peak detection.

    Args:
        fourier_transformed_activations (dict): {layer_id: np.array(num_filters, fft_length)}
        fps (float): Sampling rate in Hz (frames per second)
        threshold_factor (float): A point is considered a peak if its value is greater than threshold_factor times the average of its neighbours
    Returns:
        dict: {layer_id: {filter_id: [dominant_frequencies (Hz)]}}
    """
    dominant_frequencies = {}

    for layer_id, layer_fft in fourier_transformed_activations.items():
        num_filters, fft_length = layer_fft.shape
        # Generate frequency bins
        freqs = np.fft.fftfreq(fft_length, d=1/fps)
        freqs = np.fft.fftshift(freqs) # Center zero frequency

        dominant_frequencies[layer_id] = {}

        for filter_id in range(num_filters):
            fft_vals = layer_fft[filter_id]
            magnitudes = np.fft.fftshift(np.abs(fft_vals))

            # Remove DC component explicitly (center of fftshifted array)
            center = fft_length // 2
            magnitudes[center] = 0

            if method == 'two_neighbours':
                peak_indices = find_peaks_two_neighbours(magnitudes, threshold_factor)
            elif method == 'four_neighbours':
                peak_indices = find_peaks_four_neighbours(magnitudes, threshold_factor)
            elif method == 'snr':
                peak_indices = find_peaks_snr(magnitudes, min_snr)
            else:
                # Default to two neighbours method
                peak_indices = find_peaks_two_neighbours(magnitudes, threshold_factor)

            top_frequencies = []
            if len(peak_indices) > 0:
                # Sort peaks by magnitude (highest first)
                sorted_peaks = sorted(peak_indices, key=lambda idx: magnitudes[idx], reverse=True)
                
                # Take the top peaks (up to num_peaks)
                for i in range(min(num_peaks, len(sorted_peaks))):
                    peak_idx = sorted_peaks[i]
                    freq = abs(freqs[peak_idx])
                    top_frequencies.append(freq)
            
            # Pad with zeros if fewer than num_peaks were found
            while len(top_frequencies) < num_peaks:
                top_frequencies.append(0)
                
            # Store the top frequencies
            dominant_frequencies[layer_id][filter_id] = top_frequencies
            
    return dominant_frequencies

def save_dominant_frequencies_to_csv(dominant_frequencies, output_csv_path, image_path, gif_frequency1, gif_frequency2):
    """
    Saves dominant frequencies to CSV file, handling multiple peaks per filter.
    """
    file_exists = os.path.exists(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write an empty row
        writer.writerow([])
        # Then write the header row for the table data if file doesn't exist
        if not file_exists:
            header = ["Image", "Layer ID", "Filter ID"]
            # Add column for each peak
            num_peaks = 3
            for i in range(num_peaks):
                header.append(f"Peak {i+1} Freq")
            header.extend(["GIF Frequency 1", "GIF Frequency 2", "Flag"])
            writer.writerow(header)
            
        # Set harmonic detection parameters
        harmonic_tolerance = 1
        
        for layer_id in sorted(dominant_frequencies.keys()):
            filters = dominant_frequencies[layer_id]
            for filter_id in sorted(filters.keys()):
                peak_frequencies = filters[filter_id]  # List of top frequencies
                
                # Generate harmonics for both frequencies
                harmonics_freq1 = [n * gif_frequency1 for n in range(1, 11)]
                harmonics_freq2 = [n * gif_frequency2 for n in range(1, 11)]
                all_harmonics = harmonics_freq1 + harmonics_freq2
                
                # Check if any peak is a harmonic (ignoring zeros)
                is_harmonic = any(
                    any(abs(peak - h) < harmonic_tolerance for h in all_harmonics)
                    for peak in peak_frequencies if peak > 0
                )
                
                flag = "Same" if is_harmonic else "Different"
                
                if flag == "Different":
                    # Format row data
                    row = [image_path, layer_id, filter_id]
                    # Add all peak frequencies with formatting
                    for peak in peak_frequencies:
                        row.append(f"{peak:.10f}" if peak > 0 else "0")
                    row.extend([gif_frequency1, gif_frequency2, flag])
                    writer.writerow(row)