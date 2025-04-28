#!/usr/bin/env python3
import csv
import sys
import os
import argparse
from collections import defaultdict

def calculate_different_percentage(csv_file, layer_id=None, filter_id=None, image_key=None, detailed=False):
    """
    Calculate the percentage of 'Different' flags against all entries for a specific
    layer ID, filter ID, and/or image key.

    Args:
        csv_file (str): Path to the CSV file containing the data
        layer_id (int, optional): Specific layer ID to filter by
        filter_id (int, optional): Specific filter ID to filter by
        image_key (str, optional): Specific image key to filter by
        detailed (bool): Whether to provide detailed breakdown by filter ID

    Returns:
        dict: Dictionary containing counts and percentages
    """
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist.")
        return None

    different_count = 0
    same_count = 0

    # For detailed analysis by filter ID
    filter_stats = defaultdict(lambda: {'different': 0, 'same': 0})

    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)

        # Skip empty rows and find the header row
        header = None
        for row in reader:
            if not row:  # Skip empty rows
                continue
            if "Image" in row and "Layer ID" in row and "Filter ID" in row and "Flag" in row:
                header = row
                break

        if not header:
            print("Error: Could not find header row in CSV file.")
            return None

        # Get column indices
        try:
            image_idx = header.index("Image")
            layer_idx = header.index("Layer ID")
            filter_idx = header.index("Filter ID")
            flag_idx = header.index("Flag")
        except ValueError as e:
            print(f"Error: Missing required column in CSV: {e}")
            return None

        # Process data rows
        for row in reader:
            if not row:  # Skip empty rows
                continue

            # Apply filters if specified
            if layer_id is not None and str(layer_id) != row[layer_idx]:
                continue
            if filter_id is not None and str(filter_id) != row[filter_idx]:
                continue
            if image_key is not None and image_key not in row[image_idx]:
                continue

            # Count flags
            current_filter_id = row[filter_idx]

            if row[flag_idx] == "Different":
                different_count += 1
                filter_stats[current_filter_id]['different'] += 1
            elif row[flag_idx] == "Same":
                same_count += 1
                filter_stats[current_filter_id]['same'] += 1

    total_count = different_count + same_count
    different_percentage = (different_count / total_count * 100) if total_count > 0 else 0
    same_percentage = (same_count / total_count * 100) if total_count > 0 else 0

    result = {
        "different_count": different_count,
        "same_count": same_count,
        "total_count": total_count,
        "different_percentage": different_percentage,
        "same_percentage": same_percentage
    }

    # Add detailed filter stats if requested
    if detailed:
        filter_details = []
        for filter_id, counts in sorted(filter_stats.items(), key=lambda x: int(x[0])):
            total = counts['different'] + counts['same']
            if total > 0:
                diff_pct = (counts['different'] / total) * 100
                filter_details.append({
                    'filter_id': filter_id,
                    'different': counts['different'],
                    'same': counts['same'],
                    'total': total,
                    'different_percentage': diff_pct
                })
        result['filter_details'] = filter_details

    return result

def main():
    parser = argparse.ArgumentParser(description='Calculate percentage of Different flags in CSV data')
    parser.add_argument('--csv', type=str, default='dominant_frequencies_2n.csv',
                        help='Path to the CSV file (default: dominant_frequencies_2n.csv)')
    parser.add_argument('--layer-id', type=int, help='Filter by specific layer ID')
    parser.add_argument('--filter-id', type=int, help='Filter by specific filter ID')
    parser.add_argument('--image-key', type=str, help='Filter by image key (substring match)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed breakdown by filter ID')
    parser.add_argument('--top', type=int, default=10, help='Show only top N filters with highest different percentage')
    parser.add_argument('--all', action='store_true', help='Show all filters in detailed view (overrides --top)')

    args = parser.parse_args()

    result = calculate_different_percentage(
        args.csv,
        layer_id=args.layer_id,
        filter_id=args.filter_id,
        image_key=args.image_key,
        detailed=args.detailed
    )

    if result:
        filter_desc = []
        if args.layer_id is not None:
            filter_desc.append(f"Layer ID: {args.layer_id}")
        if args.filter_id is not None:
            filter_desc.append(f"Filter ID: {args.filter_id}")
        if args.image_key is not None:
            filter_desc.append(f"Image Key: {args.image_key}")

        filter_str = ", ".join(filter_desc) if filter_desc else "All data"

        print(f"\nResults for {filter_str}:")
        print(f"Total entries: {result['total_count']}")
        print(f"Different count: {result['different_count']} ({result['different_percentage']:.2f}%)")
        print(f"Same count: {result['same_count']} ({result['same_percentage']:.2f}%)")

        # Display detailed filter breakdown if requested
        if args.detailed and 'filter_details' in result and args.layer_id is not None and args.filter_id is None:
            # Sort by different percentage (highest first)
            sorted_details = sorted(result['filter_details'], key=lambda x: x['different_percentage'], reverse=True)

            # Filter to only show filters with different percentage > 0 or if all filters are requested
            if not args.all:
                non_zero_details = [d for d in sorted_details if d['different_percentage'] > 0]
                if non_zero_details:
                    sorted_details = non_zero_details

            # Limit to top N if specified and not showing all
            if not args.all and args.top > 0 and len(sorted_details) > args.top:
                display_details = sorted_details[:args.top]
                print(f"\nTop {args.top} filters with highest 'Different' percentage:")
            else:
                display_details = sorted_details
                if args.all:
                    print("\nAll filters sorted by 'Different' percentage:")
                else:
                    print("\nAll filters with 'Different' percentage > 0:")

            if not display_details:
                print("No filters with 'Different' flags found.")
                return

            print("-" * 60)
            print(f"{'Filter ID':<10} {'Different':<10} {'Same':<10} {'Total':<10} {'Different %':<10}")
            print("-" * 60)

            for detail in display_details:
                print(f"{detail['filter_id']:<10} {detail['different']:<10} {detail['same']:<10} {detail['total']:<10} {detail['different_percentage']:.2f}%")

            if not args.all and len(sorted_details) > args.top:
                print(f"\n...and {len(sorted_details) - args.top} more filters (use --all to see all)")

if __name__ == "__main__":
    main()
