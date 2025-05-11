#!/usr/bin/env python3
"""
Script to test the new scoring algorithm against stored FFT data.
"""

import argparse
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.database import db
from src.analysis.frequency_similarity import (
    calculate_frequency_similarity_score,
    calculate_frequency_similarity_score_v2,
    get_similarity_category,
    get_similarity_category_v2
)

def main():
    parser = argparse.ArgumentParser(description="Test new scoring algorithm against stored FFT data")
    parser.add_argument("--output", default="results/score_comparison", help="Output directory for analysis plots")
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of samples to process")
    parser.add_argument("--save-scores", action="store_true", help="Save new scores to database")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize the database
    db.init_db()
    
    # Record start time
    start_time = time.time()
    
    print(f"Testing new scoring algorithm against stored FFT data...")
    
    # Get connection
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Get runs with target frequencies
    cursor.execute(
        "SELECT id, image_id, gif_frequency1, gif_frequency2, fps FROM runs "
        "WHERE gif_frequency1 IS NOT NULL AND gif_frequency2 IS NOT NULL "
        "ORDER BY id LIMIT ?",
        (args.limit,)
    )
    
    runs = cursor.fetchall()
    
    if not runs:
        print("No runs with target frequencies found in the database.")
        conn.close()
        return
    
    print(f"Found {len(runs)} runs with target frequencies.")
    
    # Store old and new scores
    old_scores = []
    new_scores = []
    score_details = []
    
    # Process each run
    for run_idx, run in enumerate(runs):
        run_id = run['id']
        gif_frequency1 = run['gif_frequency1']
        gif_frequency2 = run['gif_frequency2']
        fps = run['fps']
        
        # Get FFT results for this run
        cursor.execute(
            "SELECT layer_id, filter_id, fft_data FROM fft_results "
            "WHERE run_id = ? ORDER BY layer_id, filter_id",
            (run_id,)
        )
        
        fft_results = cursor.fetchall()
        
        if not fft_results:
            print(f"No FFT results found for run {run_id}.")
            continue
        
        print(f"Processing run {run_id} ({run_idx+1}/{len(runs)}): {len(fft_results)} FFT results")
        
        # Process each FFT result
        for result in fft_results:
            layer_id = result['layer_id']
            filter_id = result['filter_id']
            fft_data = np.frombuffer(result['fft_data'])
            
            # Generate frequency bins based on FPS
            freqs = np.fft.fftfreq(len(fft_data), d=1/fps)
            
            # Get only positive frequencies
            positive_mask = freqs > 0
            positive_freqs = freqs[positive_mask]
            positive_magnitudes = np.abs(fft_data)[positive_mask]
            
            # Calculate old score
            old_score, _ = calculate_frequency_similarity_score(
                frequencies=positive_freqs,
                magnitudes=positive_magnitudes,
                target_freq1=gif_frequency1,
                target_freq2=gif_frequency2
            )
            
            # Calculate new score
            new_score, details = calculate_frequency_similarity_score_v2(
                frequencies=positive_freqs,
                magnitudes=positive_magnitudes,
                target_freq1=gif_frequency1,
                target_freq2=gif_frequency2
            )
            
            # Store scores
            old_scores.append(old_score)
            new_scores.append(new_score)
            
            # Store details for analysis
            score_details.append({
                'run_id': run_id,
                'layer_id': layer_id,
                'filter_id': filter_id,
                'old_score': old_score,
                'new_score': new_score,
                'target_energy': details['target_energy'],
                'entropy_score': details['entropy_score'],
                'peak_alignment': details['peak_alignment'],
                'peak_concentration': details['peak_concentration']
            })
            
            # Save new score to database if requested
            if args.save_scores:
                new_category = get_similarity_category_v2(new_score)
                cursor.execute(
                    "UPDATE dominant_frequencies SET similarity_score = ?, similarity_category = ? "
                    "WHERE run_id = ? AND layer_id = ? AND filter_id = ?",
                    (new_score, new_category, run_id, layer_id, filter_id)
                )
        
        # Commit after each run
        if args.save_scores:
            conn.commit()
    
    # Close connection
    conn.close()
    
    # Convert to numpy arrays
    old_scores = np.array(old_scores)
    new_scores = np.array(new_scores)
    
    # Print statistics
    print("\nScore Statistics:")
    print(f"Number of samples: {len(old_scores)}")
    print("\nOld Scoring Algorithm:")
    print(f"  Mean: {np.mean(old_scores):.4f}")
    print(f"  Median: {np.median(old_scores):.4f}")
    print(f"  Std Dev: {np.std(old_scores):.4f}")
    print(f"  Min: {np.min(old_scores):.4f}")
    print(f"  Max: {np.max(old_scores):.4f}")
    
    print("\nNew Scoring Algorithm:")
    print(f"  Mean: {np.mean(new_scores):.4f}")
    print(f"  Median: {np.median(new_scores):.4f}")
    print(f"  Std Dev: {np.std(new_scores):.4f}")
    print(f"  Min: {np.min(new_scores):.4f}")
    print(f"  Max: {np.max(new_scores):.4f}")
    
    # Test for normality
    _, p_old = stats.shapiro(old_scores[:1000])  # Shapiro-Wilk test (limited to 1000 samples)
    _, p_new = stats.shapiro(new_scores[:1000])
    
    print("\nNormality Test (Shapiro-Wilk):")
    print(f"  Old scores p-value: {p_old:.6f} ({'Normal' if p_old > 0.05 else 'Not normal'})")
    print(f"  New scores p-value: {p_new:.6f} ({'Normal' if p_new > 0.05 else 'Not normal'})")
    
    # Plot histograms
    plt.figure(figsize=(12, 10))
    
    # Old scores
    plt.subplot(2, 2, 1)
    plt.hist(old_scores, bins=30, alpha=0.7, color='blue')
    plt.title('Old Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # New scores
    plt.subplot(2, 2, 2)
    plt.hist(new_scores, bins=30, alpha=0.7, color='green')
    plt.title('New Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # QQ plot for old scores
    plt.subplot(2, 2, 3)
    stats.probplot(old_scores, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Old Scores)')
    plt.grid(True, alpha=0.3)
    
    # QQ plot for new scores
    plt.subplot(2, 2, 4)
    stats.probplot(new_scores, dist="norm", plot=plt)
    plt.title('Q-Q Plot (New Scores)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'score_distribution_comparison.png'))
    plt.close()
    
    # Plot score components
    plt.figure(figsize=(12, 10))
    
    components = ['target_energy', 'entropy_score', 'peak_alignment', 'peak_concentration']
    
    for i, component in enumerate(components):
        values = [detail[component] for detail in score_details]
        
        plt.subplot(2, 2, i+1)
        plt.hist(values, bins=30, alpha=0.7)
        plt.title(f'{component} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'score_components_distribution.png'))
    plt.close()
    
    # Plot correlation between old and new scores
    plt.figure(figsize=(10, 8))
    plt.scatter(old_scores, new_scores, alpha=0.5)
    plt.title('Correlation between Old and New Scores')
    plt.xlabel('Old Score')
    plt.ylabel('New Score')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(old_scores, new_scores)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=plt.gca().transAxes)
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'r--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'score_correlation.png'))
    plt.close()
    
    print(f"\nAnalysis complete! Results saved to {args.output}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()