import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# Define the path to your h5 files
base_path = '/Users/keshubai/Downloads/BRATS_1/BraTS2020_training_data/content/data/'
volume = 'volume_9'  # Only processing the first volume

# Function to get all slice files for a given volume
def get_slice_files(volume_name):
    slice_files = []
    i = 0
    while True:
        slice_file = os.path.join(base_path, f"{volume_name}_slice_{i}.h5")
        if not os.path.exists(slice_file):
            break  # Stop if no more slices are found
        slice_files.append(slice_file)
        i += 1
    return slice_files

slice_files = get_slice_files(volume)
slices = []

# Load each slice in the volume
for slice_file in slice_files:
    with h5py.File(slice_file, 'r') as f:
        # Use the correct dataset name for the image data
        dataset_name = 'image'  # Using the image dataset
        slice_data = f[dataset_name][:]
        slices.append(slice_data)

# Convert the list of slices to a 4D numpy array (assuming the last dimension is channels)
volume_data = np.stack(slices, axis=0)

# Set up the figure and animation
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')  # Set the figure background color to black
ax.set_facecolor('black')  # Set the axes background color to black
ax.axis('off')  # Hide the axes

def update_slice(frame):
    ax.clear()
    ax.imshow(volume_data[frame, :, :], cmap='gray', vmin=0, vmax=255)  # Display the slice in grayscale
    ax.set_title(f"Slice {frame + 1}", color='white')  # Set the title color to white
    ax.axis('off')  # Hide the axes for better visualization

anim = FuncAnimation(fig, update_slice, frames=volume_data.shape[0], repeat=False)

# Save the animation as an MP4
writer = FFMpegWriter(fps=10)
output_path = os.path.join(base_path, f"{volume}_black.mp4")
anim.save(output_path, writer=writer)

print(f"Saved animation for {volume} as {output_path}")

plt.close()
