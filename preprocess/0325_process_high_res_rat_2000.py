import os
import scipy.io
import numpy as np
import hashlib
from PIL import Image
from tqdm import tqdm

# Define input files and the target output directory
mat_files = [
    '/home/data1/musong/workspace/python/spen_recons_2025/nxz/our_data_1/RAT_train_1000_CP120.mat',
    '/home/data1/musong/workspace/python/spen_recons_2025/nxz/our_data_1/RAT_train_1000_CP300.mat'
]
output_dir = '/home/data1/musong/workspace/python/spen_recons/data/0325_rat/hr'
mat_key = 'ImagAll'

# 1. Create the output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 2. Setup tracking for duplicates and naming
seen_hashes = set()
saved_count = 0
duplicate_count = 0

print(f"Saving images to: {output_dir}\n")

for file_path in mat_files:
    filename = os.path.basename(file_path)
    print(f"\nLoading: {filename}...")
    
    # Load the .mat file
    try:
        data = scipy.io.loadmat(file_path)
        img_array = data[mat_key]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue
        
    print(f"Original shape in {filename}: {img_array.shape}")
    
    # Identify the 'N' axis (number of images). 
    n_axis = np.argmax(img_array.shape)
    
    # Move the N axis to the front so shape becomes (N, H, W) for easy iteration
    img_array = np.moveaxis(img_array, n_axis, 0)
    print(f"Reordered shape to iterate over: {img_array.shape}")
    print(f"Data type: {img_array.dtype}")
    
    # 3. Iterate through every image with a progress bar
    for i in tqdm(range(img_array.shape[0]), desc=f"Processing {filename}", unit="img"):
        img_complex = img_array[i]
        
        # --- NEW: Get the magnitude of the complex data ---
        img_mag = np.abs(img_complex)
        
        # Normalize the magnitude image to 0-255 (Grayscale uint8)
        img_min = np.min(img_mag)
        img_max = np.max(img_mag)
        
        if img_max == img_min:
            # Handle completely blank images to avoid division by zero
            img_uint8 = np.zeros_like(img_mag, dtype=np.uint8)
        else:
            img_uint8 = ((img_mag - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
            
        # Create a hash of the image bytes to check for duplicates
        img_hash = hashlib.md5(img_uint8.tobytes()).hexdigest()
        
        if img_hash in seen_hashes:
            duplicate_count += 1
            continue  # Skip saving if it's a duplicate
            
        # If it's unique, register the hash and save the image
        seen_hashes.add(img_hash)
        saved_count += 1
        
        # Save as PNG
        file_name = f"{saved_count}.png"
        save_path = os.path.join(output_dir, file_name)
        
        # Use PIL to save the array as a grayscale PNG
        pil_img = Image.fromarray(img_uint8, mode='L')
        pil_img.save(save_path)

print("\n--- Process Complete ---")
print(f"Total unique magnitude images saved: {saved_count}")
print(f"Total duplicates skipped: {duplicate_count}")