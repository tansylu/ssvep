"""
Test module for signal_processing.py
"""

import unittest
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.signal_processing import perform_fourier_transform, find_dominant_frequencies, is_harmonic_frequency, HarmonicType

class TestSignalProcessing(unittest.TestCase):
    """Test cases for signal processing functions"""
    
    def test_perform_fourier_transform(self):
        """Test the FFT function with a simple sine wave"""
        # Create a simple sine wave with known frequency
        fps = 30
        duration = 5
        frequency = 5  # 5 Hz
        t = np.linspace(0, duration, duration * fps)
        signal = np.sin(2 * np.pi * frequency * t)
        
        # Perform FFT
        freqs, magnitudes = perform_fourier_transform(signal, fps)
        
        # Find the index of the maximum magnitude
        max_idx = np.argmax(magnitudes)
        detected_freq = freqs[max_idx]
        
        # The detected frequency should be close to the input frequency
        self.assertAlmostEqual(detected_freq, frequency, delta=0.5)
    
    def test_find_dominant_frequencies(self):
        """Test finding dominant frequencies in a spectrum"""
        # Create a spectrum with known peaks
        freqs = np.linspace(0, 15, 100)
        magnitudes = np.zeros_like(freqs)
        
        # Add peaks at 5 Hz and 10 Hz
        peak1_idx = np.abs(freqs - 5).argmin()
        peak2_idx = np.abs(freqs - 10).argmin()
        
        magnitudes[peak1_idx] = 10
        magnitudes[peak2_idx] = 5
        
        # Find dominant frequencies
        peaks = find_dominant_frequencies(freqs, magnitudes, num_peaks=2)
        
        # Check that we found the correct peaks
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(peaks[0], 5, delta=0.5)
        self.assertAlmostEqual(peaks[1], 10, delta=0.5)
    
    def test_is_harmonic_frequency(self):
        """Test harmonic frequency detection"""
        # Test with simple harmonics
        freq1 = 5
        freq2 = 7
        
        # Test exact harmonics
        self.assertTrue(is_harmonic_frequency([10], freq1, freq2, HarmonicType.FREQ1))  # 2*freq1
        self.assertTrue(is_harmonic_frequency([14], freq1, freq2, HarmonicType.FREQ2))  # 2*freq2
        
        # Test with tolerance
        self.assertTrue(is_harmonic_frequency([10.2], freq1, freq2, HarmonicType.FREQ1, tolerance=0.5))
        
        # Test non-harmonics
        self.assertFalse(is_harmonic_frequency([11], freq1, freq2, HarmonicType.FREQ1, tolerance=0.5))
        
        # Test intermodulation
        self.assertTrue(is_harmonic_frequency([12], freq1, freq2, HarmonicType.INTERMOD))  # freq1 + freq2
        self.assertTrue(is_harmonic_frequency([2], freq1, freq2, HarmonicType.INTERMOD))   # freq2 - freq1

if __name__ == '__main__':
    unittest.main()
