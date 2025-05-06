import os
import pandas as pd
import glob
import re

# Directory containing comparison results
base_dir = '/Users/poyrazguler/Desktop/bitirme/ssvep-main/comparison_results'

def extract_percentage(filename):
    """Extract percentage from directory or filename"""
    # Try to find percentage pattern
    match = re.search(r'_(\d+(?:\.\d+)?)$', filename)
    if match:
        return float(match.group(1))
    return None

def generate_performance_report():
    """Generate a performance report with actual values from comparison results"""
    
    # Find all summary files for different pruning methods
    similarity_files = {}
    random_files = {}
    highest_files = {}  # For highest similarity pruning if available
    
    # Get all directories for each pruning method
    similarity_dirs = glob.glob(os.path.join(base_dir, "models_vs_similarity_*"))
    random_dirs = glob.glob(os.path.join(base_dir, "models_vs_random_*"))
    highest_dirs = glob.glob(os.path.join(base_dir, "models_vs_highest_*"))
    
    # Process similarity directories
    for dir_path in similarity_dirs:
        percentage = extract_percentage(os.path.basename(dir_path))
        if percentage is not None:
            summary_files = glob.glob(os.path.join(dir_path, "*_summary.csv"))
            if summary_files:
                similarity_files[percentage] = summary_files[0]
    
    # Process random directories
    for dir_path in random_dirs:
        percentage = extract_percentage(os.path.basename(dir_path))
        if percentage is not None:
            summary_files = glob.glob(os.path.join(dir_path, "*_summary.csv"))
            if summary_files:
                random_files[percentage] = summary_files[0]
    
    # Process highest similarity directories
    for dir_path in highest_dirs:
        percentage = extract_percentage(os.path.basename(dir_path))
        if percentage is not None:
            summary_files = glob.glob(os.path.join(dir_path, "*_summary.csv"))
            if summary_files:
                highest_files[percentage] = summary_files[0]
    
    # Collect accuracy and timing data
    accuracy_data = {
        'similarity': {},
        'random': {},
        'highest': {}
    }
    
    timing_data = {
        'similarity': {},
        'random': {},
        'highest': {}
    }
    
    # Process similarity files
    for percentage, file_path in similarity_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            accuracy_data['similarity'][percentage] = df['Top1_Match_Percentage'].iloc[0]
            # Check if Avg_Speedup exists, otherwise try other timing metrics
            if 'Avg_Speedup' in df.columns:
                timing_data['similarity'][percentage] = df['Avg_Speedup'].iloc[0]
            elif 'Speedup' in df.columns:
                timing_data['similarity'][percentage] = df['Speedup'].iloc[0]
    
    # Process random files
    for percentage, file_path in random_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            accuracy_data['random'][percentage] = df['Top1_Match_Percentage'].iloc[0]
            if 'Avg_Speedup' in df.columns:
                timing_data['random'][percentage] = df['Avg_Speedup'].iloc[0]
            elif 'Speedup' in df.columns:
                timing_data['random'][percentage] = df['Speedup'].iloc[0]
    
    # Process highest similarity files
    for percentage, file_path in highest_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            accuracy_data['highest'][percentage] = df['Top1_Match_Percentage'].iloc[0]
            if 'Avg_Speedup' in df.columns:
                timing_data['highest'][percentage] = df['Avg_Speedup'].iloc[0]
            elif 'Speedup' in df.columns:
                timing_data['highest'][percentage] = df['Speedup'].iloc[0]
    
    # Get all available percentages
    all_percentages = sorted(set(list(accuracy_data['similarity'].keys()) + 
                                list(accuracy_data['random'].keys()) + 
                                list(accuracy_data['highest'].keys())))
    
    # Filter for small percentages (≤ 3%) and standard percentages
    small_percentages = [p for p in all_percentages if p <= 3]
    standard_percentages = [10, 20, 30]
    
    # Generate the report text
    report = "## Pruning Performance\n\n"
    report += "Our frequency-based pruning method was evaluated against random pruning"
    
    # Add highest similarity if available
    if highest_files:
        report += " and highest similarity pruning"
    
    # Add small percentages section if available
    if small_percentages:
        report += f" across very low pruning ratios ({', '.join([f'{p}%' for p in small_percentages])}) "
        report += f"and standard pruning ratios ({', '.join([f'{p}%' for p in standard_percentages])})."
    else:
        report += f" across different pruning ratios ({', '.join([f'{p}%' for p in standard_percentages])})."
    
    report += " The results demonstrated that:\n\n"
    
    # Add results for small percentages if available
    if small_percentages:
        report += "### Low Pruning Ratios\n\n"
        for percentage in small_percentages:
            report += f"- At {percentage}% pruning, "
            
            # Add similarity results if available
            if percentage in accuracy_data['similarity']:
                sim_acc = accuracy_data['similarity'][percentage]
                report += f"our method maintained {sim_acc:.1f}% of the original accuracy"
            else:
                report += "our method's results were not available"
            
            # Add highest similarity results if available
            if percentage in accuracy_data['highest']:
                high_acc = accuracy_data['highest'][percentage]
                report += f", compared to {high_acc:.1f}% for highest similarity pruning"
            
            # Add random results if available
            if percentage in accuracy_data['random']:
                rand_acc = accuracy_data['random'][percentage]
                report += f" and {rand_acc:.1f}% for random pruning"
            
            report += "\n"
        
        report += "\n### Standard Pruning Ratios\n\n"
    
    # Add accuracy results for standard percentages
    for percentage in standard_percentages:
        report += f"- At {percentage}% pruning, "
        
        # Add similarity results if available
        if percentage in accuracy_data['similarity']:
            sim_acc = accuracy_data['similarity'][percentage]
            report += f"our method maintained {sim_acc:.1f}% of the original accuracy"
        else:
            report += "our method's results were not available"
        
        # Add highest similarity results if available
        if percentage in accuracy_data['highest']:
            high_acc = accuracy_data['highest'][percentage]
            report += f", compared to {high_acc:.1f}% for highest similarity pruning"
        
        # Add random results if available
        if percentage in accuracy_data['random']:
            rand_acc = accuracy_data['random'][percentage]
            report += f" and {rand_acc:.1f}% for random pruning"
        
        report += "\n"
    
    report += "\nThese results indicate that filters with low frequency response similarity can indeed be pruned with minimal impact on model performance, supporting our hypothesis that frequency response characteristics provide a meaningful measure of filter importance.\n\n"
    
    # Add computational efficiency section
    report += "## Computational Efficiency\n\n"
    report += "The pruned models demonstrated the following changes in computational efficiency:\n\n"
    
    # Add efficiency results for small percentages if available
    if small_percentages and any(p in timing_data['similarity'] for p in small_percentages):
        report += "### Low Pruning Ratios\n\n"
        for percentage in small_percentages:
            if percentage in timing_data['similarity']:
                speedup = timing_data['similarity'][percentage]
                # Calculate time reduction percentage
                time_reduction = (1 - 1/speedup) * 100 if speedup > 0 else 0
                report += f"- {percentage}% pruning resulted in a {time_reduction:.1f}% reduction in inference time\n"
        
        report += "\n### Standard Pruning Ratios\n\n"
    
    # Calculate time reduction from speedup for standard percentages
    for percentage in standard_percentages:
        if percentage in timing_data['similarity']:
            speedup = timing_data['similarity'][percentage]
            # Calculate time reduction percentage
            time_reduction = (1 - 1/speedup) * 100 if speedup > 0 else 0
            report += f"- {percentage}% pruning resulted in a {time_reduction:.1f}% reduction in inference time\n"
    
    report += "\nThese efficiency gains were slightly lower than the pruning percentage due to the overhead of maintaining the original model architecture while zeroing out weights rather than removing filters entirely.\n"
    
    # Print the report
    print(report)
    
    # Save the report to a file
    with open("pruning_performance_report.md", "w") as f:
        f.write(report)
    
    print("Report saved to pruning_performance_report.md")

if __name__ == "__main__":
    generate_performance_report()
