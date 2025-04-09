import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image
from scipy.spatial.distance import euclidean
from scipy.ndimage import binary_fill_holes, label, center_of_mass
sys.path.append("/home/xgq/Desktop/HF/yunshi/sam")
from .segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from .deal_segments import resolve_overlaps, calculate_centroids, show_different, \
    show_bg_segments, is_touching_edge, bounding_box_aspect_ratio, closest_aspect_ratio, touching_edge_nums, remove_noises, is_shape_convex

def show_seg(segment):
    seg = np.array(segment * 255, dtype=np.uint8)
    seg_rgb = np.stack([seg] * 3, axis=-1)
    Image.fromarray(seg_rgb).show()

def show_segment_list(segments):
    for segment in segments:
        show_seg(segment)

def segment_crater(img):
    sam_checkpoint = "/home/xgq/Desktop/HF/yunshi/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)

    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask[mask['segmentation']] = 1

    unmasked_area = np.where(combined_mask == 0, 1, 0).astype(np.uint8)
    if unmasked_area.any():
        num_labels, labels_im = cv2.connectedComponents(unmasked_area)
        for label in range(1, num_labels):
            area_mask = (labels_im == label).astype(np.uint8)
            area = np.sum(area_mask)
            M = cv2.moments(area_mask)
            centroid_x = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            centroid_y = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
            masks.append({'segmentation': area_mask, 'area': area, 'centroid': (centroid_x, centroid_y)})

    return masks

def extract_segments(w, h, sorted_masks, min_area_threshold):
    segments, centroids, areas = [], [], []
    for mask in sorted_masks:
        segmentation = mask['segmentation']
        segments.append(segmentation)
        centroids.append(center_of_mass(segmentation))
        areas.append(mask['area'])

    filtered_segments_or, filtered_centroids, filtered_areas = [], [], []
    outside_segments, noise_segments = [], []

    for segment, centroid, area in zip(segments, centroids, areas):
        if area >= min_area_threshold:
            if 0.2 * h <= centroid[0] <= 0.8 * h and 0.2 * w <= centroid[1] <= 0.8 * w and touching_edge_nums(segment) <= 3:
                filtered_segments_or.append(segment)
                filtered_centroids.append(centroid)
                filtered_areas.append(area)
            else:
                outside_segments.append(segment)
        else:
            noise_segments.append(segment)

    filtered_segments = resolve_overlaps(filtered_segments_or)
    if not filtered_segments:
        print("No filtered segments!")

    wall_segments, wall_centroids, wall_areas = [], [], []
    remaining_segments, remaining_centroids, remaining_areas = [], [], []
    for segment, centroid, area in zip(filtered_segments, filtered_centroids, filtered_areas):
        if has_large_holes(segment, min_area_threshold // 5) and area >= min_area_threshold:
            wall_segments.append(segment)
            wall_centroids.append(centroid)
            wall_areas.append(area)
        else:
            remaining_segments.append(segment)
            remaining_centroids.append(centroid)
            remaining_areas.append(area)

    internal_blocks_info = [(centroid, area, segment) for centroid, area, segment in zip(remaining_segments, remaining_centroids, remaining_areas)]
    crater_wall_info = [(centroid, area, segment) for centroid, area, segment in zip(wall_segments, wall_centroids, wall_areas)]

    return internal_blocks_info, crater_wall_info, wall_segments

def has_large_holes(binary_image, hole_area_threshold):
    filled_image = binary_fill_holes(binary_image)
    holes = np.logical_xor(binary_image, filled_image)
    lbls, num_features = label(holes)
    sizes = np.bincount(lbls.ravel())[1:]
    return np.any(sizes > hole_area_threshold)

def classify_terrain(segment, area, img, buffer_size, threshold_flat, threshold_area, threshold_std):
    terrain_heights = img[segment]
    mean_height = np.mean(terrain_heights)
    std_height = np.std(terrain_heights)
    buffered_segment = create_buffer(segment, buffer_size)
    buffer_heights = img[buffered_segment & ~segment]
    
    if buffer_heights.size == 0:
        return "Cannot classify due to lack of buffer data"
    
    mean_buffer_height = np.mean(buffer_heights)
    height_diff = np.abs(mean_height - mean_buffer_height)
    
    if height_diff > threshold_flat and area <= threshold_area[0]:
        return "Peak" if mean_height > mean_buffer_height else "Pit"
    return "Flat" if std_height < threshold_std else "simple"

def create_buffer(segment, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(segment.astype(np.uint8), kernel, iterations=1)

def evaluate_block(centroid, area, w, h):
    center_distance = min(centroid[0], w - centroid[0]) + min(centroid[1], h - centroid[1])
    return area / (w * h) - center_distance / (w + h)
