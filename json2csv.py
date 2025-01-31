import json
import os
import re
from osgeo import gdal
from save_to_shapefile import save_to_shp
import pandas as pd

def get_geo_cord(col_nums, patch_id, x, y):  # patch_id=144
    patch_size = 1024
    x_geo = (patch_id % col_nums) * patch_size + x
    y_geo = (patch_id // col_nums) * patch_size + y
    return x_geo, y_geo

def get_patch_id(file_name):
    # 定义正则表达式模式
    pattern = r'HMC_(\d+[A-Za-z]+\d+)_co5_(\d+)'
    #pattern = r'HMC_(\d+[A-Za-z]+\d+)_dt5_(\d+)'
    # 使用 re 模块的 findall 函数进行匹配
    matches = re.findall(pattern, file_name)
    # 输出匹配结果
    img_name, patch_id = matches[0]
    return patch_id

result_json_path = '/home/xgq/Desktop/HF/yunshi/results/no_dcn.json'
tiff_path = '/home/xgq/Desktop/HF/yunshi/data/test_files/'
dem_tiff_path = '/home/xgq/Desktop/HF/yunshi/data/test_files_dem/'
ref_path = '/home/xgq/Desktop/HF/yunshi/data/test_files/HMC_11W30_co5.tif'
rgb_test_path = '/home/xgq/Desktop/HF/yunshi/data/coco_rgb/val2017/'
dem_test_path = '/home/xgq/Desktop/HF/yunshi/data/coco_dem_or/val2017'
output_path = '/home/xgq/Desktop/HF/yunshi/results/shps/no_dcn.shp'
crater_list = []

with open(result_json_path, 'r') as file:
    loaded_dict = json.load(file)

tile_names = []
tile_array_list = []
geo_cords = []

for tile in sorted(os.listdir(tiff_path)):
    tile_names.append(tile)  # HMC_11W30_co5.tif
    tile = gdal.Open(tiff_path + tile)
    trans = tile.GetGeoTransform()
    X_geo_lu = int(trans[0])
    Y_geo_lu = int(trans[3])
    tile_array = tile.ReadAsArray()
    tile_array_list.append(tile_array)
    geo_cords.append([X_geo_lu, Y_geo_lu])

# print("tile_names:", tile_names)

for file in os.listdir(rgb_test_path):
    if file.endswith('.jpg'):

        #tile_name = file.split('.jpg')[0][:-4] + '.tif'
        tile_name = file.split('.jpg')[0][:13] + '.tif'

        # print("tile_name:", tile_name)

        if tile_name in tile_names:
            # print("tile_name in tile_names")
            tile_id = tile_names.index(tile_name)
        tile_lu_x, tile_lu_y = geo_cords[tile_id]

        img_name = file.split('.')[0]
        # print("img_name:", img_name)
        if img_name in loaded_dict:
            bboxes = loaded_dict[img_name]
            patch_id = int(get_patch_id(img_name))
            crater_num = len(bboxes)
            for i in range(crater_num):
                bbox = bboxes[i]
                x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                diam = int((bbox[2] + bbox[3]) / 2)
                x = (x_min + x_max) // 2
                y = (y_min + y_max) // 2
                x_tile, y_tile = get_geo_cord(13, patch_id, x, y)
                crater_list.append([tile_lu_x+(x_tile*50), tile_lu_y-(y_tile*50), diam])
        else:
            print(0)

csv_path = '/home/xgq/Desktop/HF/yunshi/results/no_dcn.csv'
c_array_big_df = pd.DataFrame(crater_list, columns=['x', 'y', 'diam'])
c_array_big_df.to_csv(csv_path, index=False)

save_to_shp(ref_path, output_path, crater_list)



