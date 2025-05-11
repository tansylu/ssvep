import numpy as np

def calculate_frequency_similarity_score(frequencies, magnitudes, target_freq1, target_freq2,
                                    max_harmonic=12, tolerance=1.0, weight_decay=0.8,
                                    include_intermodulation=True, normalization='none'):
    """
    Calculate harmonic similarity score (0-1) based on expected harmonics of two base frequencies.
    Higher score means more similar to target frequencies.

    Args:
        frequencies: List or np.array of frequency bins (Hz)
        magnitudes: List or np.array of corresponding magnitudes
        target_freq1: First target frequency
        target_freq2: Second target frequency
        max_harmonic: Maximum harmonic number to consider
        tolerance: Frequency tolerance around harmonics (Hz)
        weight_decay: Decay factor for weighting higher harmonics
        include_intermodulation: Whether to also consider intermodulation products
        normalization: Normalization method ('none', 'sqrt', 'log', 'zscore', 'robust')

    Returns:
        float: Similarity score between 0 (completely different) and 1 (perfect match)
        dict: Debug info about matched energies
    """
    # Normalize magnitudes to reduce impact of overall activation strength
    if np.sum(magnitudes) == 0:
        return 0.0, {'error': 'Zero magnitudes'}
    
    # Apply pre-normalization to magnitudes
    normalized_magnitudes = magnitudes.copy()
    if normalization == 'sqrt':
        normalized_magnitudes = np.sqrt(normalized_magnitudes)
    elif normalization == 'log':
        # Add small constant to avoid log(0)
        normalized_magnitudes = np.log1p(normalized_magnitudes)
    elif normalization == 'zscore':
        # Z-score normalization
        mean = np.mean(normalized_magnitudes)
        std = np.std(normalized_magnitudes)
        if std > 0:
            normalized_magnitudes = (normalized_magnitudes - mean) / std
    elif normalization == 'robust':
        # Robust scaling using percentiles
        p25 = np.percentile(normalized_magnitudes, 25)
        p75 = np.percentile(normalized_magnitudes, 75)
        iqr = p75 - p25
        if iqr > 0:
            normalized_magnitudes = (normalized_magnitudes - p25) / iqr
    
    # Re-normalize to sum to 1 (convert to probability distribution)
    total_energy = np.sum(normalized_magnitudes)
    if total_energy > 0:
        normalized_magnitudes = normalized_magnitudes / total_energy
    
    # Build target frequencies (harmonics + optionally intermodulation products)
    target_components = []
    harmonic_energy = 0.0
    weight_sum = 0.0

    # Harmonics of target_freq1
    for n in range(1, max_harmonic + 1):
        target_components.append({
            'frequency': n * target_freq1,
            'weight': weight_decay ** (n-1),
            'type': 'harmonic1'
        })

    # Harmonics of target_freq2
    for n in range(1, max_harmonic + 1):
        target_components.append({
            'frequency': n * target_freq2,
            'weight': weight_decay ** (n-1),
            'type': 'harmonic2'
        })

    if include_intermodulation:
        for i in range(1, 6):
            for j in range(1, 6):
                if i == 1 and j == 1:
                    continue
                sum_freq = i * target_freq1 + j * target_freq2
                diff_freq = abs(i * target_freq1 - j * target_freq2)
                
                # Higher order intermodulation products get lower weights
                intermod_weight = weight_decay ** (i+j-1)
                
                target_components.append({
                    'frequency': sum_freq,
                    'weight': intermod_weight,
                    'type': 'intermod_sum'
                })
                if diff_freq > 0:
                    target_components.append({
                        'frequency': diff_freq,
                        'weight': intermod_weight,
                        'type': 'intermod_diff'
                    })

    matched_details = []
    harmonic_contributions = {}

    for comp in target_components:
        freq = comp['frequency']
        weight = comp['weight']
        comp_type = comp['type']

        # Find bins near the target frequency
        idx = np.where(np.abs(frequencies - freq) <= tolerance)[0]
        if len(idx) > 0:
            energy_here = np.sum(normalized_magnitudes[idx])
            harmonic_energy += energy_here * weight
            
            # Track contribution by type
            if comp_type not in harmonic_contributions:
                harmonic_contributions[comp_type] = 0
            harmonic_contributions[comp_type] += energy_here * weight
            
            matched_details.append({
                'target_frequency': freq,
                'matched_energy': energy_here,
                'weight': weight,
                'type': comp_type
            })
        weight_sum += weight

    # Calculate final score with improved normalization
    if weight_sum == 0:
        final_score = 0.0
    else:
        # Normalize by weight sum to account for different numbers of components
        final_score = harmonic_energy / weight_sum
        
        # Apply sigmoid transformation to spread out the middle range
        # This helps create a more normal distribution
        final_score = 1.0 / (1.0 + np.exp(-10 * (final_score - 0.5)))

    return final_score, {
        'matched_details': matched_details,
        'harmonic_energy': harmonic_energy,
        'total_energy': total_energy,
        'weight_sum': weight_sum,
        'harmonic_contributions': harmonic_contributions
    }



def get_similarity_category(score):
    """
    Convert a similarity score to a descriptive category using a more balanced distribution.
    """
    if score >= 0.85:
        return "Very Similar"
    elif score >= 0.65:
        return "Similar"
    elif score >= 0.45:
        return "Moderately Similar"
    elif score >= 0.25:
        return "Somewhat Different"
    else:
        return "Very Different"

def calculate_frequency_similarity_score_v2(frequencies, magnitudes, target_freq1, target_freq2,
                                         max_harmonic=12, tolerance=1.0, weight_decay=0.8,
                                         include_intermodulation=True):
    """
    Calculate harmonic similarity score with a new algorithm designed to produce more normally distributed scores.
    
    This version uses a different approach:
    1. Calculates spectral concentration around target frequencies
    2. Measures spectral entropy to quantify randomness
    3. Evaluates peak alignment with expected harmonics
    4. Combines these metrics in a way that tends toward normal distribution
    
    Args:
        frequencies: List or np.array of frequency bins (Hz)
        magnitudes: List or np.array of corresponding magnitudes
        target_freq1: First target frequency
        target_freq2: Second target frequency
        max_harmonic: Maximum harmonic number to consider
        tolerance: Frequency tolerance around harmonics (Hz)
        weight_decay: Decay factor for weighting higher harmonics
        include_intermodulation: Whether to also consider intermodulation products
        
    Returns:
        float: Similarity score between 0 (completely different) and 1 (perfect match)
        dict: Debug info about the calculation components
    """
    import numpy as np
    from scipy import stats
    
    # Handle empty or invalid inputs
    if len(frequencies) == 0 or len(magnitudes) == 0:
        return 0.0, {'error': 'Empty input'}
    
    if np.sum(magnitudes) == 0:
        return 0.0, {'error': 'Zero magnitudes'}
    
    # Normalize magnitudes to sum to 1 (convert to probability distribution)
    norm_magnitudes = magnitudes / np.sum(magnitudes)
    
    # 1. Generate target frequencies (harmonics + intermodulation products)
    target_freqs = []
    
    # Harmonics of target_freq1
    for n in range(1, max_harmonic + 1):
        target_freqs.append(n * target_freq1)
    
    # Harmonics of target_freq2
    for n in range(1, max_harmonic + 1):
        target_freqs.append(n * target_freq2)
    
    # Intermodulation products
    if include_intermodulation:
        for i in range(1, 6):
            for j in range(1, 6):
                if i == 1 and j == 1:
                    continue  # Skip the base frequencies already included
                
                sum_freq = i * target_freq1 + j * target_freq2
                diff_freq = abs(i * target_freq1 - j * target_freq2)
                
                if sum_freq not in target_freqs:
                    target_freqs.append(sum_freq)
                
                if diff_freq > 0 and diff_freq not in target_freqs:
                    target_freqs.append(diff_freq)
    
    # 2. Calculate spectral concentration around target frequencies
    target_energy = 0.0
    for freq in target_freqs:
        # Find bins near the target frequency
        idx = np.where(np.abs(frequencies - freq) <= tolerance)[0]
        if len(idx) > 0:
            target_energy += np.sum(norm_magnitudes[idx])
    
    # 3. Calculate spectral entropy (measure of randomness)
    # A pure sine wave has low entropy, random noise has high entropy
    epsilon = 1e-10  # Small constant to avoid log(0)
    spectral_entropy = -np.sum(norm_magnitudes * np.log2(norm_magnitudes + epsilon))
    
    # Normalize entropy to 0-1 range (assuming max entropy is log2(N))
    max_entropy = np.log2(len(frequencies))
    if max_entropy > 0:
        normalized_entropy = spectral_entropy / max_entropy
    else:
        normalized_entropy = 1.0
    
    # 4. Find peaks in the spectrum
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(norm_magnitudes, height=0.01)
    
    # 5. Calculate peak alignment score
    peak_freqs = frequencies[peaks]
    peak_alignment = 0.0
    
    if len(peak_freqs) > 0:
        for freq in peak_freqs:
            # Find closest target frequency
            closest_target = min(target_freqs, key=lambda x: abs(x - freq))
            
            # Calculate distance as a fraction of tolerance
            distance = abs(freq - closest_target) / tolerance
            
            # Score decreases with distance
            if distance <= 1.0:
                peak_alignment += (1.0 - distance)
        
        # Normalize by number of peaks
        peak_alignment = peak_alignment / len(peak_freqs)
    
    # 6. Calculate peak height concentration
    # This measures if the energy is concentrated in a few strong peaks (good)
    # or spread across many small peaks (bad)
    if len(peaks) > 0:
        peak_heights = norm_magnitudes[peaks]
        peak_concentration = np.sum(peak_heights**2) / (np.sum(peak_heights)**2) * len(peak_heights)
    else:
        peak_concentration = 0.0
    
    # 7. Combine metrics into final score
    # Weight the components to produce a more normal distribution
    w1, w2, w3, w4 = 0.4, 0.2, 0.3, 0.1  # Weights for each component
    
    # Invert entropy (lower entropy is better)
    entropy_score = 1.0 - normalized_entropy
    
    # Combine scores
    raw_score = (
        w1 * target_energy + 
        w2 * entropy_score + 
        w3 * peak_alignment + 
        w4 * peak_concentration
    )
    
    # Apply sigmoid transformation to spread scores more evenly
    # This helps create a more normal distribution
    final_score = 1.0 / (1.0 + np.exp(-5 * (raw_score - 0.5)))
    
    # Return score and debug info
    return final_score, {
        'target_energy': target_energy,
        'entropy_score': entropy_score,
        'peak_alignment': peak_alignment,
        'peak_concentration': peak_concentration,
        'raw_score': raw_score
    }

def get_similarity_category_v2(score):
    """
    Convert a similarity score to a descriptive category using thresholds designed
    for the new scoring algorithm to produce a more balanced distribution.
    """
    if score >= 0.80:
        return "Very Similar"
    elif score >= 0.60:
        return "Similar"
    elif score >= 0.40:
        return "Moderately Similar"
    elif score >= 0.20:
        return "Somewhat Different"
    else:
        return "Very Different"
