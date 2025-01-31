import numpy as np
from skimage import measure
from skimage.measure import label, regionprops
from demo import get_segments, show_anns
from sam.deal_segments import is_touching_edge, is_shape_convex
from PIL import Image
from sam.segment_crater import segment_crater, extract_segments, evaluate_block
from dem_enhance import enhance_dem_array
import cv2

def sam_pos_opt(sam_seg_path, tile_dem, tile_dom, tile_w, tile_h, x_geo_min, y_geo_min, x_geo_max, y_geo_max, diam):

    tile_dem = np.stack([tile_dem] * 3, axis=-1)
    cropped_array = tile_dem[int(y_geo_min):int(y_geo_max), int(x_geo_min):int(x_geo_max), :]
    cropped_array = enhance_dem_array(cropped_array, n=2)
    cropped_array = np.stack([cropped_array] * 3, axis=-1)

    h, w = cropped_array.shape[:2]

    floor_seg, largest_wall_region, crater_segment, centroid_x, centroid_y = get_segments(cropped_array, sam_seg_path)

    if crater_segment is None:
        return diam, centroid_x, centroid_y, None, None, cropped_array, None, None, None, None


    num = 0
    while is_touching_edge(crater_segment) and num < 3:
        x_geo_min, y_geo_min, x_geo_max, y_geo_max, floor_seg, largest_wall_region, \
        crater_segment, centroid_x, centroid_y = expand_bbox(tile_dem, tile_w, tile_h, w,
                                                             h, x_geo_min, x_geo_max, y_geo_min,
                                                             y_geo_max, centroid_x, centroid_y, sam_seg_path)
        num += 1

    if crater_segment is None:
        return diam, centroid_x, centroid_y, None, None, cropped_array, None, None, None, None

    bbox = find_bounding_box(crater_segment)

    if bbox is None:
        return diam, centroid_x, centroid_y, None, None, None, None, None, None, None

    x_min, y_min, x_max, y_max = bbox

    x_min_cls = max(round(x_geo_min + x_min - w*0.15), 0)
    y_min_cls = max(round(y_geo_min + y_min - h*0.15), 0)
    x_max_cls = min(round(x_geo_min + x_max + w*0.15), tile_w)
    y_max_cls = min(round(y_geo_min + y_max + h*0.15), tile_h)

    #dem_crater_area = tile_or_unit8[y_min_cls:y_max_cls, x_min_cls:x_max_cls, :]
    dem_crater_area = tile_dem[y_min_cls:y_max_cls, x_min_cls:x_max_cls, 0]
    dem_crater_area_255 = scale_to_ubyte(dem_crater_area)
    dem_crater_area_enhanced = enhance_dem_array(dem_crater_area, n=2)
    dem_crater_area_enhanced = np.stack([dem_crater_area_enhanced] * 3, axis=-1)
    dom_crater_area = tile_dom[:, y_min_cls:y_max_cls, x_min_cls:x_max_cls]

    return diam, centroid_x, centroid_y, dem_crater_area, dem_crater_area_255, dem_crater_area_enhanced, dom_crater_area, floor_seg, largest_wall_region, crater_segment

def expand_bbox(tile_dem, tile_w, tile_h, w, h, x_geo_min, x_geo_max, y_geo_min, y_geo_max, centroid_x, centroid_y, sam_seg_path):

    x_bia = centroid_x - w // 2
    y_bia = centroid_y - h // 2
    # print("x_bia, y_bia:", x_bia, y_bia)
    if x_bia >= 0 and y_bia >= 0:
        x_min_n = x_geo_min
        y_min_n = y_geo_min
        x_max_n = min(round(x_geo_max + x_bia), tile_w)
        y_max_n = min(round(y_geo_max + y_bia), tile_h)
    elif x_bia >= 0 and y_bia < 0:
        x_min_n = x_geo_min
        y_min_n = max(round(y_geo_min + y_bia), 0)
        x_max_n = min(round(x_geo_max + x_bia), tile_h)
        y_max_n = y_geo_max
    elif x_bia < 0 and y_bia >= 0:
        x_min_n = max(round(x_geo_min + x_bia), 0)
        y_min_n = y_geo_min
        x_max_n = x_geo_max
        y_max_n = min(round(y_geo_max + y_bia), tile_h)
    elif x_bia < 0 and y_bia < 0:
        x_min_n = max(round(x_geo_min + x_bia), 0)
        y_min_n = max(round(y_geo_min + y_bia), 0)
        x_max_n = x_geo_max
        y_max_n = y_geo_max

    if y_min_n == 0:
        y_max_n = y_min_n + h
    elif y_max_n == tile_h:
        y_min_n = y_max_n - h

    if x_min_n == 0:
        x_max_n = x_min_n + w
    elif x_max_n == tile_w:
        x_min_n = x_max_n - w

    # print("tile_or_unit8.shape:", tile_or_unit8.shape[:2])
    # print("x_min_n, y_min_n, x_max_n, y_max_n, tile_h, tile_w:", x_min_n, y_min_n, x_max_n, y_max_n, tile_h, tile_w)
    print("expanding......")
    new_cropped_array = tile_dem[y_min_n:y_max_n, x_min_n:x_max_n]
    new_cropped_array = enhance_dem_array(new_cropped_array, n=2)
    new_cropped_array = np.stack([new_cropped_array] * 3, axis=-1)
    floor_seg, largest_wall_region, crater_segment, centroid_x, centroid_y = get_segments(new_cropped_array, sam_seg_path)
    return x_min_n, y_min_n, x_max_n, y_max_n, floor_seg, largest_wall_region, crater_segment, centroid_x, centroid_y

def find_bounding_box(binary_array):
    rows = np.any(binary_array, axis=1)
    cols = np.any(binary_array, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return min_col, min_row, max_col + 1, max_row + 1

def find_largest_bounding_box_and_segment(binary_array):
    # 标记连通区域
    labels = measure.label(binary_array, connectivity=2)
    props = measure.regionprops(labels)
    if not props:
        return None, None
    largest_region = max(props, key=lambda r: r.area)
    min_row, min_col, max_row, max_col = largest_region.bbox
    largest_segment = np.zeros_like(binary_array, dtype=np.uint8)
    largest_segment[labels == largest_region.label] = 1

    return  min_col, min_row, max_col, max_row, largest_segment

def show_seg(segment):
    seg = np.array(segment * 255, dtype=np.uint8)
    seg = np.stack([seg] * 3, axis=-1)
    seg_img = Image.fromarray(seg)
    seg_img.show()

def scale_to_ubyte(array):
    array_float = array.astype(np.float32)
    min_val = np.min(array_float)
    max_val = np.max(array_float)
    # 公式：new_value = ((original_value - min_val) / (max_val - min_val)) * 255
    scaled_array = ((array_float - min_val) / (max_val - min_val)) * 255
    scaled_array = np.floor(scaled_array).astype(np.uint8)
    scaled_array = np.clip(scaled_array, 0, 255)
    return scaled_array