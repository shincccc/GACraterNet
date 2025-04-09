from PIL import Image
import cv2
import os
import numpy as np
import sys
sys.path.append('/home/xgq/Desktop/HF/yunshi')
from get_terr_factors import get_terr_factors
from osgeo import gdal
from shapely.geometry import Point
import json
from pos_opt import pos_opt
import re
import pandas as pd
from segment_and_reposition import sam_pos_opt

def get_patch_id(file_name):
    """Extract patch ID from filename using regex."""
    pattern = r'HMC_(\d+[A-Za-z]+\d+)_co5_(\d+)'
    matches = re.findall(pattern, file_name)
    return matches[0][1]

def get_geo_cord(col_nums, patch_id, x, y):
    """Convert patch coordinates to geographic coordinates."""
    patch_size = 1024
    x_geo = (patch_id % col_nums) * patch_size + x
    y_geo = (patch_id // col_nums) * patch_size + y
    return x_geo, y_geo

def scale_to_ubyte(array):
    """Scale array to 0-255 range and convert to uint8."""
    array_float = array.astype(np.float32)
    min_val, max_val = np.min(array_float), np.max(array_float)
    scaled = ((array_float - min_val) / (max_val - min_val)) * 255
    return np.clip(np.floor(scaled), 0, 255).astype(np.uint8)

def bbox2atts(dom_tiles_path, dem_tiles_path, loaded_dict, model_path, rgb_test_path, dem_test_path, c1, c2, tiff_ref_path, pixel_size, shp_save_path):
    """Process bounding boxes to extract crater attributes and save results."""
    # Load tile information
    tile_array_list, geo_cords, tile_names = [], [], []
    for tile in sorted(os.listdir(dem_tiles_path)):
        tile_names.append(tile)
        tile_ds = gdal.Open(dem_tiles_path + tile)
        trans = tile_ds.GetGeoTransform()
        tile_array_list.append(tile_ds.ReadAsArray().astype(np.float32))
        geo_cords.append([int(trans[0]), int(trans[3])])

    tile_array_list2, tile_names2 = [], []
    for tile in sorted(os.listdir(dom_tiles_path)):
        tile_names2.append(tile)
        tile_ds = gdal.Open(dom_tiles_path + tile)
        tile_array_list2.append(tile_ds.ReadAsArray())

    # Process each patch
    id, pred_num = 0, 0
    for file in os.listdir(rgb_test_path):
        if not file.endswith('.jpg'):
            continue

        # Get tile info
        tile_name = file.split('_co5')[0] + '_dt5.tif'
        if tile_name not in tile_names:
            continue
        tile_id = tile_names.index(tile_name)
        tile_or_dem, tile_or_dom = tile_array_list[tile_id], tile_array_list2[tile_id]
        tile_lu_x, tile_lu_y = geo_cords[tile_id]
        tile_h, tile_w = tile_or_dem.shape[0], tile_or_dem.shape[1]
        col_nums = tile_w // 1024

        # Get patch info and bounding boxes
        img_name = file.split('.')[0]
        if img_name not in loaded_dict:
            print(f"No bbox found for {img_name}")
            continue
        bboxes = loaded_dict[img_name]
        pred_num += len(bboxes)

        for i, bbox in enumerate(bboxes):
            # Calculate coordinates and diameter
            x_min, y_min = int(bbox[0]), int(bbox[1])
            x_max, y_max = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            diam = int((bbox[2] + bbox[3]) / 2)
            patch_id = int(get_patch_id(img_name))
            x_tile_min, y_tile_min = get_geo_cord(col_nums, patch_id, x_min, y_min)
            x_tile_max, y_tile_max = get_geo_cord(col_nums, patch_id, x_max, y_max)
            x_geo, y_geo = (x_tile_min + x_tile_max) // 2, (y_tile_min + y_tile_max) // 2

            # Expand bounding box
            x_tile_min = max(int(x_tile_min - diam * 0.2), 0)
            y_tile_min = max(int(y_tile_min - diam * 0.2), 0)
            x_tile_max = min(int(x_tile_max + diam * 0.2), tile_w)
            y_tile_max = min(int(y_tile_max + diam * 0.2), tile_h)

            if diam >= 60:
                # Segment and optimize position
                sam_seg_path = f'/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/sam/{id}_{img_name}_{i}.jpg'
                diam, x, y, dem_crater_area, dem_crater_area_255, dem_crater_area_enhanced, dom_crater_area, floor_seg, \
                wall_seg, crater_seg = sam_pos_opt(sam_seg_path, tile_or_dem, tile_or_dom, tile_w, tile_h,
                                                  x_tile_min, y_tile_min, x_tile_max, y_tile_max, diam)

                if dem_crater_area is None:
                    # Fallback to basic cropping if segmentation fails
                    dem_array = tile_or_dem[y_tile_min:y_tile_max, x_tile_min:x_tile_max]
                    dem_img2 = np.stack([scale_to_ubyte(dem_array)] * 3, axis=-1)
                    cv2.imwrite(f'/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/wsb/{id}_{img_name}_{i}.jpg', dem_img2)
                    factors = get_terr_factors(dem_array, dem_img2, model_path, num_profiles=12)
                else:
                    # Use segmented crater area
                    dem_crater_area_255 = np.stack([dem_crater_area_255] * 3, axis=-1)
                    cv2.imwrite(f'/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/{id}_{img_name}_{i}.jpg', dem_crater_area_255)
                    cv2.imwrite(f'/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/enhanced/{id}_{img_name}_{i}.jpg', dem_crater_area_enhanced)
                    x_geo = tile_lu_x + (x_tile_min + x) * 50
                    y_geo = tile_lu_y - (y_tile_min + y) * 50
                    factors = get_terr_factors(dem_crater_area, dem_crater_area_255, model_path, num_profiles=8)

                # Append large crater data
                c2.append([x_geo, y_geo, diam, *factors, id, img_name])
            else:
                # Append small crater data
                c1.append([tile_lu_x + x_geo * 50, tile_lu_y - y_geo * 50, diam, id, img_name])
            id += 1

    return c1, c2

if __name__ == "__main__":
    """Main execution: process crater data and save to CSV."""
    result_json_path = '/home/xgq/Desktop/HF/yunshi/results/exp_image_data.json'
    rgb_test_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dom_clips/'
    dem_test_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dem_clips/'
    dem_tiles_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dem/'
    dom_tiles_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dom/'
    tiff_ref_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dem/HMC_13E10_dt5.tif'
    rf_model_path = '/home/xgq/Desktop/HF/yunshi/machine_learing/weights/profile_cls_model_8.joblib'
    shp_save_path = '/home/xgq/Desktop/HF/yunshi/results/poly_shp/craters.shp'
    csv1 = '/home/xgq/Desktop/HF/yunshi/results/smaller_3km_results_exp.csv'
    csv2 = '/home/xgq/Desktop/HF/yunshi/results/bigger_3km_results_exp.csv'

    # Load JSON data
    with open(result_json_path, 'r') as file:
        loaded_dict = json.load(file)

    # Process craters
    c1, c2 = [], []
    pixel_size = (50, 50)
    c1, c2 = bbox2atts(dom_tiles_path, dem_tiles_path, loaded_dict, rf_model_path, rgb_test_path, dem_test_path, c1, c2, tiff_ref_path, pixel_size, shp_save_path)

    # Save results to CSV
    pd.DataFrame(c1, columns=['x', 'y', 'diam', 'id', 'file']).to_csv(csv1, index=False)
    pd.DataFrame(c2, columns=['x', 'y', 'diam', 'depth', 'sjb', 'morf', 'peak_h', 'pit_d', 'floor_h', 
                            'floor_distance', 'floor_s_ratio', 'valley_bia', 'mid_h', 'rim_h', 'rim_w', 'id', 'file']).to_csv(csv2, index=False)
