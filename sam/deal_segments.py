import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, LineString, MultiPolygon
from shapely.ops import unary_union
from scipy.ndimage import label as label2
import cv2
from PIL import Image

def show_seg0(segment):
    seg = np.array(segment * 255, dtype=np.uint8)
    seg = np.stack([seg] * 3, axis=-1)
    Image.fromarray(seg).show()

def show_segment_list(segments):
    for segment in segments:
        show_seg0(segment)

def resolve_overlaps(segments):
    areas = [np.sum(segment) for segment in segments]
    ordered_indices = np.argsort(areas)[::-1].tolist()
    final_segments = []
    
    for index in ordered_indices:
        segment = segments[index].copy()
        for other_index in ordered_indices:
            if index == other_index:
                continue
            other_segment = segments[other_index]
            overlap = np.logical_and(segment, other_segment)
            if np.sum(overlap) == 0 or np.sum(segment) < np.sum(other_segment):
                continue
            segment[overlap] = False
        final_segments.append(segment)
    
    return final_segments

def calculate_centroids(masks):
    centroids = []
    for mask in masks:
        true_indices = np.argwhere(mask)
        if true_indices.size > 0:
            centroid_x = true_indices[:, 1].sum() / len(true_indices)
            centroid_y = true_indices[:, 0].sum() / len(true_indices)
            centroids.append((centroid_x, centroid_y))
        else:
            centroids.append(None)
    return centroids

def show_different(filtered_segments, filtered_segments_or):
    if len(filtered_segments) != len(filtered_segments_or):
        raise ValueError("Lists must have equal length")
    
    num_segments = len(filtered_segments)
    plt.figure(figsize=(10 * num_segments, 10))
    
    for i in range(num_segments):
        plt.subplot(1, num_segments * 2, i * 2 + 1)
        plt.imshow(filtered_segments[i], cmap='gray')
        plt.title(f'Processed Segment {i + 1}')
        plt.axis('off')
        
        plt.subplot(1, num_segments * 2, i * 2 + 2)
        plt.imshow(filtered_segments_or[i], cmap='gray')
        plt.title(f'Original Segment {i + 1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_bg_segments(segments):
    num_segments = len(segments)
    num_rows = int(np.ceil(np.sqrt(num_segments)))
    plt.figure(figsize=(10 * num_rows, 10 * num_rows))
    
    for i, segment in enumerate(segments):
        plt.subplot(num_rows, num_rows, i + 1)
        plt.imshow(segment, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Segment {i + 1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def is_touching_edge(segment):
    if segment is None:
        return False
    return np.any(segment[0, :]) or np.any(segment[-1, :]) or np.any(segment[:, 0]) or np.any(segment[:, -1])

def touching_edge_nums(segment):
    edges = {
        "top": segment[0:5, :],
        "bottom": segment[-5:, :],
        "left": segment[:, 0:5],
        "right": segment[:, -5:]
    }
    return sum(np.any(edge) for edge in edges.values())

def remove_noises(segment):
    labeled_array, num_features = label2(segment, structure=np.ones((3, 3)))
    if num_features == 0:
        return segment
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0
    return np.where(labeled_array == sizes.argmax(), 1, 0)

def bounding_box_aspect_ratio(segment):
    labeled_segment = label(segment)
    props = regionprops(labeled_segment)
    if props:
        bbox = props[0].bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        return width / height if height > 0 else 0
    return None

def closest_aspect_ratio(target_ratio, ratios):
    return min(range(len(ratios)), key=lambda i: abs(ratios[i] - target_ratio))

def show_seg(segment):
    plt.imshow(segment, cmap='gray')
    plt.axis('off')
    plt.show()

def is_shape_convex(segment):
    segment = remove_noises(segment)
    if segment.dtype != np.uint8:
        segment = (segment * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    
    contour = contours[0]
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    area_ratio = area / hull_area if hull_area > 0 else 0
    
    convexity = "concave" if area_ratio < 0.8 else "convex"
    x, y, w, h = cv2.boundingRect(hull)
    bounding_box = ((x, y), (x + w, y + h))
    
    return convexity, hull, bounding_box
