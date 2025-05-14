import pandas as pd
import argparse
import os
import random
import numpy as np

def generate_bottom_filters_from_csv(csv_path, percentage, output_file=None):
    """
    Generate a list of layer and filter IDs representing the bottom X percent of filters
    based on a CSV file containing filter statistics.

    Args:
        csv_path: Path to the CSV file containing filter statistics
        percentage: Percentage of filters to select (0-100)
        output_file: Path to save the output (if None, prints to console)

    Returns:
        List of tuples (layer_id, filter_id) representing the bottom X percent of filters
    """
    # Read the CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Check if required columns exist
    required_columns = ['Layer', 'Filter', 'Total Similarity Score']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain column: {col}")

    # Sort by the score column (assuming lower score means less important)
    df = df.sort_values(by='Total Similarity Score', ascending=True)

    # Calculate how many filters to select
    total_filters = len(df)
    num_filters_to_select = int(total_filters * percentage / 100)

    print(f"Total filters: {total_filters}")
    print(f"Selecting bottom {percentage}% ({num_filters_to_select} filters)")

    # Select the bottom X percent
    bottom_filters = df.head(num_filters_to_select)

    # Extract layer and filter IDs
    selected_filters = []
    for _, row in bottom_filters.iterrows():
        layer_id = int(row['Layer'])
        filter_id = int(row['Filter'])
        selected_filters.append((layer_id, filter_id))

    # Sort by layer ID and then filter ID for better readability
    selected_filters.sort()

    # Output the results
    if output_file:
        write_filters_to_file(selected_filters, output_file)
    else:
        for layer_id, filter_id in selected_filters:
            print(f"{layer_id},{filter_id}")

    return selected_filters

def generate_random_filters_from_csv(csv_path, percentage, output_file=None, seed=None):
    """
    Generate a list of randomly selected layer and filter IDs
    based on a CSV file containing filter statistics.

    Args:
        csv_path: Path to the CSV file containing filter statistics
        percentage: Percentage of filters to select (0-100)
        output_file: Path to save the output (if None, prints to console)
        seed: Random seed for reproducibility

    Returns:
        List of tuples (layer_id, filter_id) representing randomly selected filters
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Read the CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Check if required columns exist
    required_columns = ['Layer', 'Filter']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain column: {col}")

    # Calculate how many filters to select
    total_filters = len(df)
    num_filters_to_select = int(total_filters * percentage / 100)

    print(f"Total filters: {total_filters}")
    print(f"Randomly selecting {percentage}% ({num_filters_to_select} filters)")

    # Randomly select filters
    random_indices = np.random.choice(total_filters, num_filters_to_select, replace=False)
    random_filters = df.iloc[random_indices]

    # Extract layer and filter IDs
    selected_filters = []
    for _, row in random_filters.iterrows():
        layer_id = int(row['Layer'])
        filter_id = int(row['Filter'])
        selected_filters.append((layer_id, filter_id))

    # Sort by layer ID and then filter ID for better readability
    selected_filters.sort()

    # Output the results
    if output_file:
        write_filters_to_file(selected_filters, output_file)
    else:
        for layer_id, filter_id in selected_filters:
            print(f"{layer_id},{filter_id}")

    return selected_filters

def create_specific_filters_list():
    """
    Create a list with the specific predefined layer and filter IDs.

    Returns:
        List of tuples (layer_id, filter_id)
    """
    # The specific list of layer and filter IDs
    filters = [
        (13, 215), (13, 11), (13, 85), (13, 148), (13, 44),
        (14, 64), (14, 86), (14, 175), (14, 112), (14, 154), (14, 0), (14, 123), (14, 242),
        (14, 131), (14, 153), (14, 138), (14, 252), (14, 23), (14, 124), (14, 60), (14, 28),
        (14, 164), (14, 224),
        (15, 70), (15, 115), (15, 40), (15, 446), (15, 85), (15, 367), (15, 486), (15, 409),
        (15, 444), (15, 322), (15, 337), (15, 66), (15, 302), (15, 413), (15, 20), (15, 119),
        (15, 192), (15, 247), (15, 152), (15, 142), (15, 366), (15, 414), (15, 279), (15, 51),
        (15, 141), (15, 324), (15, 403), (15, 433), (15, 43), (15, 100), (15, 404), (15, 345),
        (15, 173), (15, 361), (15, 307), (15, 274), (15, 321), (15, 487), (15, 464), (15, 63),
        (15, 447), (15, 174), (15, 478), (15, 118), (15, 144), (15, 380), (15, 438), (15, 75),
        (15, 328), (15, 65), (15, 145), (15, 227), (15, 214), (15, 223), (15, 235), (15, 176),
        (15, 125), (15, 111), (15, 442), (15, 405), (15, 502), (15, 242), (15, 200), (15, 349),
        (15, 389), (15, 109), (15, 133), (15, 11), (15, 28), (15, 282), (15, 295), (15, 265),
        (15, 428), (15, 167), (15, 382), (15, 286), (15, 451), (15, 355), (15, 3), (15, 445),
        (15, 207), (15, 219), (15, 273), (15, 148), (15, 408), (15, 183), (15, 465), (15, 210),
        (15, 369),
        (16, 223), (16, 227), (16, 45), (16, 330), (16, 380), (16, 210), (16, 238), (16, 471),
        (16, 280), (16, 225), (16, 285), (16, 324), (16, 94), (16, 468), (16, 64), (16, 472),
        (16, 21), (16, 262), (16, 508), (16, 95), (16, 436), (16, 187), (16, 38), (16, 148),
        (16, 25), (16, 183), (16, 8), (16, 305), (16, 48), (16, 11), (16, 354), (16, 494),
        (16, 385), (16, 339), (16, 188), (16, 499), (16, 316), (16, 279), (16, 407), (16, 57),
        (16, 432), (16, 231), (16, 106), (16, 489), (16, 69), (16, 453), (16, 298), (16, 49),
        (16, 465), (16, 371), (16, 123), (16, 215), (16, 287), (16, 88), (16, 350), (16, 482),
        (16, 36), (16, 393), (16, 71),
        (18, 20), (18, 411), (18, 90), (18, 135), (18, 33), (18, 329), (18, 206), (18, 438),
        (18, 4), (18, 396), (18, 421)
    ]
    return filters

def write_filters_to_file(filters, output_file):
    """
    Write a list of filters to a file.

    Args:
        filters: List of tuples (layer_id, filter_id)
        output_file: Path to save the output
    """
    # Count filters by layer
    layer_counts = {}
    for layer_id, _ in filters:
        if layer_id not in layer_counts:
            layer_counts[layer_id] = 0
        layer_counts[layer_id] += 1

    # Print summary
    print(f"Total filters: {len(filters)}")
    for layer_id, count in sorted(layer_counts.items()):
        print(f"Layer {layer_id}: {count} filters")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Write to file
    with open(output_file, 'w') as f:
        for layer_id, filter_id in filters:
            f.write(f"{layer_id},{filter_id}\n")

    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate a list of filters to prune')
    parser.add_argument('--csv', type=str, help='Path to the CSV file containing filter statistics')
    parser.add_argument('--percentage', type=float, default=10.0,
                        help='Percentage of filters to select (default: 10.0)')
    parser.add_argument('--output', type=str, help='Path to save the output (if not specified, a name will be generated)')
    parser.add_argument('--use-predefined', action='store_true',
                        help='Use the predefined list of filters instead of generating from CSV')
    parser.add_argument('--random', action='store_true',
                        help='Randomly select filters instead of using similarity scores')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Generate output filename if not specified
    if args.output is None:
        if args.use_predefined:
            output_file = 'data/predefined_filters.txt'
        else:
            # Create a descriptive filename
            prefix = "random_" if args.random else ""
            percentage_str = f"{int(args.percentage)}_percent"
            output_file = f'data/filters_to_prune/{prefix}filters_to_prune_{percentage_str}.txt'
    else:
        output_file = args.output

    if args.use_predefined:
        # Use the predefined list of filters
        filters = create_specific_filters_list()
        write_filters_to_file(filters, output_file)
    else:
        # Generate from CSV
        if not args.csv:
            parser.error("--csv is required when not using --use-predefined")

        # Validate percentage
        if args.percentage <= 0 or args.percentage >= 100:
            raise ValueError("Percentage must be between 0 and 100")

        if args.random:
            # Generate random filters
            generate_random_filters_from_csv(args.csv, args.percentage, output_file, args.seed)
        else:
            # Generate the bottom filters based on similarity scores
            generate_bottom_filters_from_csv(args.csv, args.percentage, output_file)

if __name__ == "__main__":
    main()
