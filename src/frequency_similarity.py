import numpy as np

def calculate_frequency_similarity_score(frequencies, magnitudes, target_freq1, target_freq2,
                                    max_harmonic=12, tolerance=1.0, weight_decay=0.8,
                                    include_intermodulation=True):
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

    Returns:
        float: Similarity score between 0 (completely different) and 1 (perfect match)
        dict: Debug info about matched energies
    """
    total_energy = np.sum(magnitudes)
    harmonic_energy = 0.0
    weight_sum = 0.0

    # Build target frequencies (harmonics + optionally intermodulation products)
    target_components = []

    # Harmonics of target_freq1
    for n in range(1, max_harmonic + 1):
        target_components.append({
            'frequency': n * target_freq1,
            'weight': weight_decay ** (n-1)
        })

    # Harmonics of target_freq2
    for n in range(1, max_harmonic + 1):
        target_components.append({
            'frequency': n * target_freq2,
            'weight': weight_decay ** (n-1)
        })

    if include_intermodulation:
        for i in range(1, 6):
            for j in range(1, 6):
                if i == 1 and j == 1:
                    continue
                sum_freq = i * target_freq1 + j * target_freq2
                diff_freq = abs(i * target_freq1 - j * target_freq2)
                target_components.append({
                    'frequency': sum_freq,
                    'weight': weight_decay ** (i+j-1)
                })
                if diff_freq > 0:
                    target_components.append({
                        'frequency': diff_freq,
                        'weight': weight_decay ** (i+j-1)
                    })

    matched_details = []

    for comp in target_components:
        freq = comp['frequency']
        weight = comp['weight']

        # Find bins near the target frequency
        idx = np.where(np.abs(frequencies - freq) <= tolerance)[0]
        if len(idx) > 0:
            energy_here = np.sum(magnitudes[idx])
            harmonic_energy += energy_here * weight
            matched_details.append({
                'target_frequency': freq,
                'matched_energy': energy_here,
                'weight': weight
            })
        weight_sum += weight

    # Normalize
    if total_energy == 0 or weight_sum == 0:
        final_score = 0.0
    else:
        final_score = harmonic_energy / (total_energy * weight_sum)

    return final_score, {
        'matched_details': matched_details,
        'harmonic_energy': harmonic_energy,
        'total_energy': total_energy,
        'weight_sum': weight_sum
    }



def get_similarity_category(score):
    """
    Convert a similarity score to a descriptive category.
    """
    if score >= 0.8:
        return "Very Similar"
    elif score >= 0.6:
        return "Similar"
    elif score >= 0.4:
        return "Moderately Similar"
    elif score >= 0.2:
        return "Somewhat Different"
    else:
        return "Very Different"
