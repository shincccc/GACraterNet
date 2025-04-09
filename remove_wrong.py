from demo import get_segments
import numpy as np
import os
from sam.deal_segments import is_touching_edge
from PIL import Image

def remove_fp(pd_path, tp_path, fp_path, sam_seg_path):
    """Filter false positives from predicted DEM images and save results."""
    for pic in os.listdir(pd_path):
        # Load and enhance DEM image
        img_path = os.path.join(pd_path, pic)
        c_array = np.array(Image.open(img_path))
        c_array_eh = enhance_dem_array(c_array, 3)

        # Segment the enhanced array
        floor_seg, largest_wall_region, crater_segment, centroid_x, \
        centroid_y = get_segments(c_array_eh, os.path.join(sam_seg_path, pic))

        # Save as true positive or false positive based on segmentation
        processed_img = Image.fromarray(c_array_eh)
        if crater_segment is not None:
            processed_img.save(os.path.join(tp_path, pic))
        else:
            processed_img.save(os.path.join(fp_path, pic))

def enhance_dem_array(array, n):
    """Enhance DEM array by modulating height values and normalizing to 0-255."""
    # Ensure input is 2D
    if len(array.shape) != 2:
        array = array[:, :, 0]

    # Calculate height range and modulation factor
    max_val, min_val = np.max(array), np.min(array)
    delta_h = max_val - min_val
    a = delta_h / n

    # Modulate and normalize array
    new_array = array % a
    norm_array = (new_array - np.min(new_array)) / (np.max(new_array) - np.min(new_array)) * 255
    norm_array = norm_array.astype(np.uint8)
    return np.stack([norm_array] * 3, axis=-1)

if __name__ == "__main__":
    """Main execution: process DEM images to remove false positives."""
    pd_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/exp_dem'
    tp_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/tp/'
    fp_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/fp/'
    sam_seg_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/sam/'
    remove_fp(pd_path, tp_path, fp_path, sam_seg_path)
