from osgeo import gdal
import numpy as np
from scipy.ndimage import label, find_objects, center_of_mass
from skimage.measure import regionprops
import pandas as pd
from PIL import Image
import cv2
from get_terr_factors import get_terr_factors
from skimage.segmentation import clear_border
import csv
import os
import matplotlib.pyplot as plt

def scale_to_ubyte(array):
    """Scale array to 0-255 range and convert to uint8."""
    array_float = array.astype(np.float32)
    min_val, max_val = np.min(array_float), np.max(array_float)
    scaled = ((array_float - min_val) / (max_val - min_val)) * 255
    return np.clip(np.floor(scaled), 0, 255).astype(np.uint8)

def extract_region(dem_region, d, swin_model_path, id, file_name):
    """Extract terrain factors from DEM region and save as image."""
    dem_region_uint8 = scale_to_ubyte(dem_region)
    dem_region_uint8 = np.stack([dem_region_uint8] * 3, axis=-1)
    img = Image.fromarray(dem_region_uint8)
    img.save(f"/home/xgq/Desktop/HF/yunshi/results/fld_results/{file_name}_fld_{id}.jpg")
    return get_terr_factors(dem_region, dem_region_uint8, swin_model_path, num_profiles=12)

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = bbox1
    x1b, y1b, x2b, y2b = bbox2
    inter_x1, inter_y1 = max(x1, x1b), max(y1, y1b)
    inter_x2, inter_y2 = min(x2, x2b), min(y2, y2b)
    intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2b - x1b) * (y2b - y1b)
    union = bbox1_area + bbox2_area - intersection
    return intersection / union if union > 0 else 0

def get_best_region_bbox(dataset, area_threshold, density_threshold, iou_threshold=0.3):
    """Identify best regions in image based on area and density, filter by IoU."""
    b_image = dataset.ReadAsArray().astype(np.uint16)
    b_image[b_image == 32768] = 0
    b_image = scale_to_ubyte(b_image)
    _, binary_image = cv2.threshold(b_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = clear_border(binary_image)
    labeled_array, _ = label(binary_image)
    regions = regionprops(labeled_array)

    valid_objects = [(r.bbox, r.area) for r in regions 
                    if r.area >= area_threshold and 
                    r.area / ((r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])) > density_threshold]

    # Visualize valid regions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(b_image, cmap='gray')
    for bbox, _ in valid_objects:
        minr, minc, maxr, maxc = bbox
        ax.add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr, 
                                  fill=False, edgecolor='red', linewidth=2))
    plt.title("Valid Region Bboxes")
    plt.axis('off')
    plt.show()

    # Filter overlapping regions
    best_bboxes = []
    for bbox, area in sorted(valid_objects, key=lambda x: x[1], reverse=True):
        if all(calculate_iou(bbox, prev_bbox) < iou_threshold for prev_bbox in best_bboxes):
            best_bboxes.append(bbox)

    # Visualize filtered regions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(b_image, cmap='gray')
    for bbox in best_bboxes:
        minr, minc, maxr, maxc = bbox
        ax.add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr, 
                                  fill=False, edgecolor='blue', linewidth=2))
    plt.title("Best Region Bboxes after IOU Filtering")
    plt.axis('off')
    plt.show()

    return best_bboxes, b_image, labeled_array

def get_region_bbox(c3, gcc_tiff_path, dem_tiff_path, swin_model_path, rf_model_path, file_name, 
                   area_threshold=40000, density_threshold=0.1):
    """Process regions from GCC and DEM data, extract terrain factors."""
    dataset = gdal.Open(gcc_tiff_path)
    geotransform = dataset.GetGeoTransform()
    origin_x, origin_y = geotransform[0], geotransform[3]
    dem_dataset = gdal.Open(dem_tiff_path).ReadAsArray()
    objects, _, _ = get_best_region_bbox(dataset, area_threshold, density_threshold)

    for i, obj in enumerate(objects):
        x_bbox, y_bbox, w, h = obj[1], obj[0], abs(obj[3] - obj[1]), abs(obj[2] - obj[0])
        d = int((w + h) / 2 * 1.2)
        xc, yc = w // 2, h // 2

        if d >= 60 and abs(w / h) >= 0.40:
            dem_region = dem_dataset[obj[0] - int(d * 0.2):obj[2] + int(d * 0.2), 
                                   obj[1] - int(d * 0.2):obj[3] + int(d * 0.2)]
            if dem_region.size > 0:
                factors = extract_region(dem_region, d, swin_model_path, i, file_name)
                c3.append([int(origin_x + (x_bbox + xc) * 50), int(origin_y - (y_bbox + yc) * 50), 
                          d, *factors, f"{i}_fld", file_name])
    return c3

if __name__ == "__main__":
    """Main execution: process TIFF files and save results to CSV."""
    gcc_path = '/home/xgq/Desktop/HF/yunshi/data/gcc1/'
    dem_path = '/home/xgq/Desktop/HF/yunshi/data/dem1/'
    rf_model_path = '/home/xgq/Desktop/HF/yunshi/machine_learing/weights/profile_cls_model_6.joblib'
    swin_model_path = '/home/xgq/Desktop/HF/yunshi/swin_transformer/weights/morf_best_model-2.pth'
    csv3 = '/home/xgq/Desktop/HF/yunshi/results/bigger_50km_results_exp.csv'

    c3 = []
    for file in os.listdir(dem_path):
        file_name = file.split('.tif')[0]
        dem_tiff_path = dem_path + file
        gcc_tiff_path = gcc_path + file_name + '_gcc.tif'
        c3 = get_region_bbox(c3, gcc_tiff_path, dem_tiff_path, swin_model_path, rf_model_path, file_name, 
                            area_threshold=40000, density_threshold=0.3)

    pd.DataFrame(c3, columns=['x', 'y', 'diam', 'depth', 'sjb', 'morf', 'peak_h', 'pit_d', 
                            'floor_h', 'floor_distance', 'floor_s_ratio', 'valley_bia', 
                            'mid_h', 'rim_h', 'rim_w', 'id', 'file']).to_csv(csv3, index=False)
