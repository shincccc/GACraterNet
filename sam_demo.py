from PIL import Image
import numpy as np
from sam.segment_crater import segment_crater, extract_segments, evaluate_block
from sam.deal_segments import remove_noises, is_touching_edge
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import rasterio
import random
from osgeo import gdal
import cv2
from dem_enhance import enhance_dem_array

def find_bounding_box(binary_array):
    """Find bounding box coordinates of non-zero region in binary array."""
    rows, cols = np.any(binary_array, axis=1), np.any(binary_array, axis=0)
    if not (np.any(rows) and np.any(cols)):
        return None
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    return min_col, min_row, max_col + 1, max_row + 1

def show_seg(segment):
    """Display binary segment as an image."""
    seg = np.array(segment * 255, dtype=np.uint8)
    seg_rgb = np.stack([seg] * 3, axis=-1)
    Image.fromarray(seg_rgb).show()

def generate_colors(anns, num_colors):
    """Generate random colors for annotations."""
    if not anns:
        return None, None
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    colors = np.random.rand(num_colors, 3)
    for i in range(3):
        colors[:, i] = (colors[:, i] - colors[:, i].min()) / (colors[:, i].max() - colors[:, i].min())
    return colors, sorted_anns

def show_anns(sorted_anns, colors, sam_seg_path):
    """Visualize annotations with colors and save to file."""
    max_shape = sorted_anns[0]['segmentation'].shape
    img = np.zeros((max_shape[0], max_shape[1], 4), dtype=np.float32)
    img[:, :, 3] = 1  # Set alpha channel to opaque
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        img[m == 1] = np.concatenate([colors[i], [0.35]])
    img_bgr = (img[:, :, :3] * 255).astype(np.uint8)
    cv2.imwrite(sam_seg_path, img_bgr)

def compute_centroid(segment):
    """Calculate centroid of a binary segment."""
    y_indices, x_indices = np.where(segment == 1)
    return (np.mean(x_indices), np.mean(y_indices)) if y_indices.size > 0 else None

def get_segments(img_array, sam_seg_path):
    """Segment image and extract floor, wall, and crater regions."""
    h, w = img_array.shape[:2]
    s = h * w
    masks = segment_crater(img_array)
    colors, sorted_anns = generate_colors(masks, len(masks))
    if colors is not None:
        show_anns(sorted_anns, colors, sam_seg_path)

    internal_blocks_info, crater_wall_info, wall_segments = extract_segments(w, h, masks, min_area_threshold=s / 10)
    if not internal_blocks_info:
        return None, None, None, w / 2, h / 2

    floor_centroid, floor_area, floor_seg = internal_blocks_info[0]
    centroid_y, centroid_x = floor_centroid
    if wall_segments:
        combined_walls = np.sum(wall_segments, axis=0).astype(np.uint8) * 255
        labels = measure.label(combined_walls, connectivity=2)
        props = measure.regionprops(labels)
        if props:
            max_area = max(prop.area for prop in props)
            max_label = next(prop.label for prop in props if prop.area == max_area)
            largest_wall_region = np.where(labels == max_label, 1, 0)
            crater_segment = np.logical_or(floor_seg, combined_walls)
        else:
            largest_wall_region, crater_segment = None, floor_seg
    else:
        largest_wall_region, crater_segment = None, floor_seg

    crater_segment = remove_noises(crater_segment)
    return floor_seg, largest_wall_region, crater_segment, centroid_x, centroid_y

def get_crs_from_tiff(tiff_path):
    """Extract CRS from TIFF file."""
    with rasterio.open(tiff_path) as src:
        return src.crs.to_string()

def preprocess_array(arr):
    """Apply binary opening and closing to smooth array."""
    struct = generate_binary_structure(2, 2)
    arr = binary_opening(arr, structure=struct).astype(int)
    return binary_closing(arr, structure=struct).astype(int)

def array_to_polygon(arr, tile_lu_x, tile_lu_y, patch_lu_x, patch_lu_y):
    """Convert binary array to simplified polygon."""
    if arr is None:
        return None
    arr = preprocess_array(arr)
    rows, cols = arr.shape
    exterior_coords = []
    grid_size = 50
    lu_x, lu_y = tile_lu_x + patch_lu_x * 50, tile_lu_y - patch_lu_y * 50

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current, neighbors = arr[i][j], [arr[i-1][j], arr[i+1][j], arr[i][j-1], arr[i][j+1]]
            if (current == 1 and sum(neighbors) == 2) or (current == 0 and sum(neighbors) == 3):
                exterior_coords.append((lu_x + j * grid_size, lu_y - i * grid_size))

    if exterior_coords:
        unique_coords = list(set(exterior_coords))
        polygon = Polygon(unique_coords).convex_hull
        return polygon.simplify(1.0, preserve_topology=False)
    return None

def get_geo_cord(col_nums, patch_id, x, y):
    """Convert patch coordinates to geographic coordinates."""
    patch_size = 1024
    return (patch_id % col_nums) * patch_size + x, (patch_id // col_nums) * patch_size + y

def expand_window(tile_array, patch_id, x_min, y_min, w0, h0, w, h, centroid_x, centroid_y):
    """Expand cropping window based on centroid offset."""
    x_bia, y_bia = centroid_x - w // 2, centroid_y - h // 2
    col_nums, tile_w, tile_h = 13, 13336, 17782
    diam = (w0 + h0) // 2
    x_tile_min, y_tile_min = get_geo_cord(col_nums, patch_id, x_min, y_min)
    x_tile_max, y_tile_max = get_geo_cord(col_nums, patch_id, x_min + w0, y_min + h0)
    x_tile_min, y_tile_min = max(int(x_tile_min - diam * 0.2), 0), max(int(y_tile_min - diam * 0.2), 0)
    x_tile_max, y_tile_max = min(int(x_tile_max + diam * 0.2), tile_w), min(int(y_tile_max + diam * 0.2), tile_h)

    # Adjust window based on centroid bias
    if x_bia >= 0 and y_bia >= 0:
        x_min_n, y_min_n = x_tile_min, y_tile_min
        x_max_n, y_max_n = min(round(x_tile_max + x_bia), tile_w), min(round(y_tile_max + y_bia), tile_h)
    elif x_bia >= 0 and y_bia < 0:
        x_min_n, y_min_n = x_tile_min, max(round(y_tile_min + y_bia), 0)
        x_max_n, y_max_n = min(round(x_tile_max + x_bia), tile_w), y_tile_max
    elif x_bia < 0 and y_bia >= 0:
        x_min_n, y_min_n = max(round(x_tile_min + x_bia), 0), y_tile_min
        x_max_n, y_max_n = x_tile_max, min(round(y_tile_max + y_bia), tile_h)
    else:
        x_min_n, y_min_n = max(round(x_tile_min + x_bia), 0), max(round(y_tile_min + y_bia), 0)
        x_max_n, y_max_n = x_tile_max, y_tile_max

    # Ensure window fits dimensions
    if y_min_n == 0: y_max_n = y_min_n + h
    elif y_max_n == tile_h: y_min_n = y_max_n - h
    if x_min_n == 0: x_max_n = x_min_n + w
    elif x_max_n == tile_w: x_min_n = x_max_n - w

    new_cropped_array = np.stack([tile_array[y_min_n:y_max_n, x_min_n:x_max_n]] * 3, axis=-1)
    return (*expand_window_helper(new_cropped_array, sam_seg_path), x_min_n, y_min_n, x_max_n, y_max_n)

def expand_window_helper(new_cropped_array, sam_seg_path):
    """Helper function to segment expanded window."""
    return get_segments(new_cropped_array, sam_seg_path)

if __name__ == "__main__":
    """Main execution: process and segment DEM image."""
    sam_seg_path = '/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/sam/73_HMC_13E10_co5_96_0.jpg'
    img_path = '/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/wsb/73_HMC_13E10_co5_96_0.jpg'

    image = Image.open(img_path)
    img_array = enhance_dem_array(np.array(image), n=5)
    img_array = np.stack([img_array] * 3, axis=-1)
    h, w = img_array.shape[:2]
    floor_seg, largest_wall_region, crater_segment, centroid_x, centroid_y = get_segments(img_array, sam_seg_path)
