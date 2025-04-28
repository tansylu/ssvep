"""
Database module for storing and retrieving filter statistics.
This module provides functions to save and query filter statistics.
"""

import db
import sqlite3
import os
from datetime import datetime

def init_filter_stats_db():
    """
    Initialize the filter statistics tables in the database.
    """
    conn = db.get_connection()
    cursor = conn.cursor()

    # Create filter_stats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS filter_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        layer_id INTEGER NOT NULL,
        filter_id INTEGER NOT NULL,
        total_similarity_score REAL NOT NULL DEFAULT 0,
        total_images INTEGER NOT NULL DEFAULT 0,
        avg_similarity_score REAL NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (layer_id, filter_id)
    )
    ''')

    # Create filter_stats_images table to track which images were processed for each filter
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS filter_stats_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filter_stats_id INTEGER NOT NULL,
        image_id INTEGER NOT NULL,
        similarity_score REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (filter_stats_id) REFERENCES filter_stats (id) ON DELETE CASCADE,
        FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE,
        UNIQUE (filter_stats_id, image_id)
    )
    ''')

    conn.commit()
    conn.close()

def update_filter_stats(filter_stats_table, image_path):
    """
    Update filter statistics in the database.

    Args:
        filter_stats_table (dict): Dictionary with filter statistics
        image_path (str): Path to the image file
    """
    # Initialize the database if it doesn't exist
    init_filter_stats_db()

    # Get or create the image record
    image_id = db.get_or_create_image(image_path)

    conn = db.get_connection()
    cursor = conn.cursor()

    # Update filter statistics for each filter
    for (layer_id, filter_id), stats in filter_stats_table.items():
        total_images = stats["total_images"]
        if total_images > 0:
            total_similarity_score = stats["total_similarity_score"]
            avg_similarity_score = total_similarity_score / total_images

            # Check if the filter stats record exists
            cursor.execute(
                "SELECT id FROM filter_stats WHERE layer_id = ? AND filter_id = ?",
                (layer_id, filter_id)
            )
            result = cursor.fetchone()

            if result:
                # Update existing record
                filter_stats_id = result['id']
                cursor.execute(
                    "UPDATE filter_stats SET total_similarity_score = ?, total_images = ?, avg_similarity_score = ?, updated_at = ? "
                    "WHERE id = ?",
                    (total_similarity_score, total_images, avg_similarity_score, datetime.now(), filter_stats_id)
                )
            else:
                # Insert new record
                cursor.execute(
                    "INSERT INTO filter_stats (layer_id, filter_id, total_similarity_score, total_images, avg_similarity_score) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (layer_id, filter_id, total_similarity_score, total_images, avg_similarity_score)
                )
                filter_stats_id = cursor.lastrowid

            # Check if this image has already been processed for this filter
            cursor.execute(
                "SELECT id FROM filter_stats_images WHERE filter_stats_id = ? AND image_id = ?",
                (filter_stats_id, image_id)
            )
            result = cursor.fetchone()

            if not result:
                # Add the image to the filter_stats_images table
                # Get the similarity score for this image (the last one added)
                similarity_score = total_similarity_score / total_images  # Use average as an approximation
                cursor.execute(
                    "INSERT INTO filter_stats_images (filter_stats_id, image_id, similarity_score) "
                    "VALUES (?, ?, ?)",
                    (filter_stats_id, image_id, similarity_score)
                )

    conn.commit()
    conn.close()

def get_filter_stats():
    """
    Get all filter statistics from the database.

    Returns:
        list: List of filter statistics
    """
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM filter_stats ORDER BY avg_similarity_score DESC"
    )
    results = cursor.fetchall()

    conn.close()

    # Convert the results to a list of dictionaries
    return [dict(row) for row in results]

def export_filter_stats_to_csv(output_path):
    """
    Export filter statistics to a CSV file.

    Args:
        output_path (str): Path to the output CSV file
    """
    import csv

    stats = get_filter_stats()

    if not stats:
        print("No filter statistics found in the database.")
        return

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Layer", "Filter", "Total Similarity Score", "Total Images", "Avg Similarity Score"])

        # Write data
        for stat in stats:
            writer.writerow([
                stat['layer_id'],
                stat['filter_id'],
                f"{stat['total_similarity_score']:.4f}",
                stat['total_images'],
                f"{stat['avg_similarity_score']:.4f}"
            ])

    print(f"Filter statistics exported to {output_path}")
