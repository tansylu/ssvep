import pandas as pd

# Load your data
df = pd.read_csv("dominant_frequencies.csv")

# Group by 'Image' and count rows
image_counts = df.groupby('Image').size()

# Divide by 120
frame_sets = image_counts / 120

# Convert to DataFrame and reset index so 'Image' is a column
frame_sets_df = frame_sets.reset_index()
frame_sets_df.columns = ['Image', 'Count_Divided_By_120']

# Save to CSV
frame_sets_df.to_csv("image_frame_sets.csv", index=False)

print("Saved as image_frame_sets.csv with image names as a column.")
