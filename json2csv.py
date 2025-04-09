import json
import os
import re
from osgeo import gdal
from save_to_shapefile import save_to_shp
import pandas as pd

def get_geo_cord(col_nums, patch_id, x, y):
    """Convert patch coordinates to geographic coordinates."""
    patch_size = 1024
    x_geo = (patch_id % col_nums) * patch_size + x
    y_geo = (patch_id // col_nums) * patch_size + y
    return x_geo, y_geo

def get_patch_id(file_name):
    """Extract patch ID from filename using regex."""
    pattern = r'HMC_(\d+[A-Za-z]+\d+)_co5_(\d+)'
    matches = re.findall(pattern, file_name)
    return matches[0][1]  # Return patch ID

# Define file paths
result_json_path = '/home/xgq/Desktop/HF/yunshi/results/no_dcn.json'
tiff_path = '/home/xgq/Desktop/HF/yunshi/data/test_files/'
dem_tiff_path = '/home/xgq/Desktop/HF/yunshi/data/test_files_dem/'
ref_path = '/home/xgq/Desktop/HF/yunshi/data/test_files/HMC_11W30_co5.tif'
rgb_test_path = '/home/xgq/Desktop/HF/yunshi/data/coco_rgb/val2017/'
dem_test_path = '/home/xgq/Desktop/HF/yunshi/data/coco_dem_or/val2017'
output_path = '/home/xgq/Desktop/HF/yunshi/results/shps/no_dcn.shp'
crater_list = []

# Load JSON results
with open(result_json_path, 'r') as file:
    loaded_dict = json.load(file)

# Collect tile information
tile_names, tile_array_list, geo_cords = [], [], []
for tile in sorted(os.listdir(tiff_path)):
    tile_names.append(tile)
    tile_ds = gdal.Open(tiff_path + tile)
    trans = tile_ds.GetGeoTransform()
    geo_cords.append([int(trans[0]), int(trans[3])])  # Upper-left corner coordinates
    tile_array_list.append(tile_ds.ReadAsArray())

# Process RGB images and extract crater data
for file in os.listdir(rgb_test_path):
    if file.endswith('.jpg'):
        tile_name = file.split('.jpg')[0][:13] + '.tif'
        if tile_name in tile_names:
            tile_id = tile_names.index(tile_name)
            tile_lu_x, tile_lu_y = geo_cords[tile_id]
            img_name = file.split('.')[0]
            
            if img_name in loaded_dict:
                bboxes = loaded_dict[img_name]
                patch_id = int(get_patch_id(img_name))
                for bbox in bboxes:
                    x_min, y_min = int(bbox[0]), int(bbox[1])
                    x_max, y_max = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                    diam = int((bbox[2] + bbox[3]) / 2)
                    x, y = (x_min + x_max) // 2, (y_min + y_max) // 2
                    x_tile, y_tile = get_geo_cord(13, patch_id, x, y)
                    crater_list.append([tile_lu_x + (x_tile * 50), tile_lu_y - (y_tile * 50), diam])
            else:
                print(0)

# Save results to CSV
csv_path = '/home/xgq/Desktop/HF/yunshi/results/no_dcn.csv'
c_array_big_df = pd.DataFrame(crater_list, columns=['x', 'y', 'diam'])
c_array_big_df.to_csv(csv_path, index=False)

# Save to shapefile
save_to_shp(ref_path, output_path, crater_list)
