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
        different INTEGER NOT NULL DEFAULT 0,
        same INTEGER NOT NULL DEFAULT 0,
        total INTEGER NOT NULL DEFAULT 0,
        diff_percent REAL NOT NULL DEFAULT 0,
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
        is_harmonic BOOLEAN NOT NULL,
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
        total = stats["different"] + stats["same"]
        if total > 0:
            diff_percent = (stats["different"] / total) * 100
            
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
                    "UPDATE filter_stats SET different = ?, same = ?, total = ?, diff_percent = ?, updated_at = ? "
                    "WHERE id = ?",
                    (stats["different"], stats["same"], total, diff_percent, datetime.now(), filter_stats_id)
                )
            else:
                # Insert new record
                cursor.execute(
                    "INSERT INTO filter_stats (layer_id, filter_id, different, same, total, diff_percent) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (layer_id, filter_id, stats["different"], stats["same"], total, diff_percent)
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
                is_harmonic = stats["same"] > stats["different"]
                cursor.execute(
                    "INSERT INTO filter_stats_images (filter_stats_id, image_id, is_harmonic) "
                    "VALUES (?, ?, ?)",
                    (filter_stats_id, image_id, is_harmonic)
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
        "SELECT * FROM filter_stats ORDER BY diff_percent DESC"
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
        writer.writerow(["Layer", "Filter", "Different", "Same", "Total", "Diff %"])
        
        # Write data
        for stat in stats:
            writer.writerow([
                stat['layer_id'],
                stat['filter_id'],
                stat['different'],
                stat['same'],
                stat['total'],
                f"{stat['diff_percent']:.2f}"
            ])
    
    print(f"Filter statistics exported to {output_path}")
