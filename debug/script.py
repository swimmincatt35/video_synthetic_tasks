import os
from PIL import Image
import numpy as np

save_root = "./rank0_video0"  # adjust this if needed

def is_white(img, tol=1e-3):
    """Check if image is all white (within tolerance)."""
    arr = np.array(img).astype(np.float32) / 255.0
    return np.allclose(arr, 1.0, atol=tol)

    
white_indices = []
for frame_name in sorted(os.listdir(save_root)):
    frame_path = os.path.join(save_root, frame_name)
    img = Image.open(frame_path).convert("RGB")
    if is_white(img):
        white_indices.append(frame_name)
    
print(f"white frames at {white_indices}")
