import os
import numpy as np
from PIL import Image

def generate_mock_images():
    """
    Generates synthetic training and validation images to test the model
    training and prediction pipeline locally.
    
    Creates:
    - 32 mock flooded images (blue squares with random noise)
    - 32 mock non-flooded images (green/brown squares with random noise)
    under the expected FloodNet dataset directory structure.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    flooded_dir = os.path.join(base_dir, "data", "Train", "Labeled", "Flooded", "image")
    non_flooded_dir = os.path.join(base_dir, "data", "Train", "Labeled", "Non-Flooded", "image")
    
    os.makedirs(flooded_dir, exist_ok=True)
    os.makedirs(non_flooded_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    
    num_samples = 32
    print(f"Generating {num_samples} mock images for Flooded class under: {flooded_dir}")
    for i in range(num_samples):
        # Create a 224x224 blue-ish image with random noise (representing water)
        img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_arr[:, :, 2] = 200  # Blue channel high
        img_arr += np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
        img.save(os.path.join(flooded_dir, f"mock_flooded_{i}.jpg"))
        
    print(f"Generating {num_samples} mock images for Non-Flooded class under: {non_flooded_dir}")
    for i in range(num_samples):
        # Create a 224x224 green-ish image with random noise (representing trees/grass)
        img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_arr[:, :, 1] = 180  # Green channel high
        img_arr += np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
        img.save(os.path.join(non_flooded_dir, f"mock_non_flooded_{i}.jpg"))
        
    print("Mock dataset generation completed successfully!")

if __name__ == "__main__":
    generate_mock_images()
