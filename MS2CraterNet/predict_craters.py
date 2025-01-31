from PIL import Image
import cv2
import os
import numpy as np
import sys
sys.path.append('/home/xgq/Desktop/HF/yunshi')  # 添加上上级目录
from get_terr_factors import get_terr_factors
from osgeo import gdal
from shapely.geometry import Point
import json
from pos_opt import pos_opt
import re
import pandas as pd
from segment_and_reposition import sam_pos_opt

def get_patch_id(file_name):
    # 定义正则表达式模式
    pattern = r'HMC_(\d+[A-Za-z]+\d+)_co5_(\d+)'
    # 使用 re 模块的 findall 函数进行匹配
    matches = re.findall(pattern, file_name)
    # 输出匹配结果
    img_name, patch_id = matches[0]
    return patch_id

def get_geo_cord(col_nums, patch_id, x, y):  # patch_id=144
    patch_size = 1024
    x_geo = (patch_id % col_nums) * patch_size + x
    y_geo = (patch_id // col_nums) * patch_size + y
    return x_geo, y_geo

def scale_to_ubyte(array):
    # 将数组转换为浮点数，以进行计算
    array_float = array.astype(np.float32)
    # 找到数组的最小值和最大值
    min_val = np.min(array_float)
    max_val = np.max(array_float)
    # 进行线性缩放，将数值范围映射到 0 到 255
    # 公式：new_value = ((original_value - min_val) / (max_val - min_val)) * 255
    scaled_array = ((array_float - min_val) / (max_val - min_val)) * 255
    # 将结果转换为整数
    scaled_array = np.floor(scaled_array).astype(np.uint8)
    # 确保数值在 0 到 255 的范围内
    scaled_array = np.clip(scaled_array, 0, 255)
    return scaled_array

def bbox2atts(dom_tiles_path, dem_tiles_path, loaded_dict, model_path, rgb_test_path, dem_test_path, c1, c2, tiff_ref_path, pixel_size, shp_save_path):
    # print("loaded_dict:", loaded_dict)
    #tile信息
    tile_array_list = []
    geo_cords = []
    tile_names = []
    # floor_seg_polys = []
    # wall_seg_polys = []
    # crater_seg_polys = []
    # crater_seg_morf = []
    # crater_seg_sjb = []
    ids = []

    for tile in sorted(os.listdir(dem_tiles_path)):
        tile_names.append(tile) #HMC_11W30_dt5.tif
        tile = gdal.Open(dem_tiles_path+tile)
        trans = tile.GetGeoTransform()
        X_geo_lu = int(trans[0])
        Y_geo_lu = int(trans[3])
        tile_array =  tile.ReadAsArray().astype(np.float32)
        tile_array_list.append(tile_array)
        geo_cords.append([X_geo_lu, Y_geo_lu])

    tile_array_list2 = []
    tile_names2 = []
    for tile in sorted(os.listdir(dom_tiles_path)):
        tile_names2.append(tile) #HMC_11W30_co5.tif
        tile = gdal.Open(dom_tiles_path+tile)
        tile_array = tile.ReadAsArray()
        tile_array_list2.append(tile_array)

    # 单张patch
    id = 0
    pred_num = 0
    for file in os.listdir(rgb_test_path):
        if file.endswith('.jpg'): #预测一个裁剪图内的撞击坑
            # print(file)
            #tile左上角坐标和其他信息
            tile_name = file.split('_co5')[0] + '_dt5.tif'

            # print("tile_name:", tile_name)
            if tile_name in tile_names:
                tile_id = tile_names.index(tile_name)
            tile_or_dem = tile_array_list[tile_id]
            tile_or_dom = tile_array_list2[tile_id]
            tile_lu_x, tile_lu_y = geo_cords[tile_id]
            tile_h = tile_or_dem.shape[0]
            tile_w = tile_or_dem.shape[1]
            col_nums = tile_w // 1024

            #patch信息
            img_name = file.split('.')[0]
            # 根据 img_name 从 loaded_dict 中获取边界框信息
            if img_name in loaded_dict:
                bboxes = loaded_dict[img_name]   #img_name = HMC_13E10_co5_10
                # print(f"Bboxes for {img_name}: {bboxes}")
            else:
                print(f"No bbox found for {img_name}")
                continue

            crater_num = len(bboxes)
            pred_num += crater_num

            for i in range(crater_num):
                bbox = bboxes[i]
                print("bbox:", bbox)
                x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
                diam = int((bbox[2]+bbox[3])/2)

                #tile坐标
                patch_id = int(get_patch_id(img_name))
                x_tile_min, y_tile_min = get_geo_cord(col_nums, patch_id, x_min, y_min)
                x_tile_max, y_tile_max = get_geo_cord(col_nums, patch_id, x_max, y_max)
                x_geo = (x_tile_min + x_tile_max)//2
                y_geo = (y_tile_min + y_tile_max)//2

                #裁剪圖片
                x_tile_min, y_tile_min, x_tile_max, y_tile_max = int(x_tile_min - int(diam * 0.2)), int(y_tile_min - int(diam * 0.2)), int(
                    x_tile_max + int(diam * 0.2)), int(y_tile_max + int(diam * 0.2))

                x_tile_min = max(x_tile_min, 0)
                y_tile_min = max(y_tile_min, 0)
                x_tile_max = min(x_tile_max, tile_w)
                y_tile_max = min(y_tile_max, tile_h)
                # print("id:", id)

                if diam >= 60:
                    # DEM
                    # x_tile, y_tile, y_tile_min, y_tile_max, x_tile_min, x_tile_max = pos_opt(tile_or_dem, tile_w,
                    #                                                                          tile_h, x_tile_min,
                    #                                                                          y_tile_min, x_tile_max,
                    #                                                                          y_tile_max)
                    #分割并转为shp格式，返回修正后的包围框
                    sam_seg_path = '/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/sam/'+ f'{id}_' + file.split('.')[0] + f'_{i}.jpg'
                    diam, x, y, dem_crater_area, dem_crater_area_255, dem_crater_area_enhanced, dom_crater_area, floor_seg, \
                    wall_seg, crater_seg = sam_pos_opt(sam_seg_path, tile_or_dem, tile_or_dom, tile_w, tile_h,
                                                       x_tile_min, y_tile_min, x_tile_max, y_tile_max, diam)
                    cv2.imwrite('/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/enhanced/' + f'{id}_' + file.split('.')[0] + f'_{i}.jpg', dem_crater_area_enhanced)

                    if dem_crater_area is None:
                        dem_array = tile_or_dem[y_tile_min:y_tile_max, x_tile_min:x_tile_max]
                        dem_array2 = scale_to_ubyte(dem_array)
                        dem_img2 = np.stack([dem_array2]*3, axis = -1)
                        cv2.imwrite('/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/wsb/' + f'{id}_' + file.split('.')[0] + f'_{i}.jpg', dem_img2)
                        depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w = get_terr_factors(dem_array, dem_img2, model_path, num_profiles=12)
                        c2.append([tile_lu_x+x_geo*50, tile_lu_y-y_geo*50, diam, depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w, id, file.split('.')[0]])
                        id += 1
                    else:
                        dem_crater_area_255 = np.stack([dem_crater_area_255] * 3, axis=-1)
                        cv2.imwrite('/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/' + f'{id}_' + file.split('.')[0] + f'_{i}.jpg', dem_crater_area_255)
                        # cv2.imwrite('/home/xgq/Desktop/HF/yunshi/results/A1A5_sam/' + f'{id}_' + file.split('.')[0] + f'_{i}.jpg', dom_crater_area.transpose(1, 2, 0))
                        cv2.imwrite('/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/enhanced/' + f'{id}_' + file.split('.')[0] + f'_{i}.jpg', dem_crater_area_enhanced)
                        x_geo = tile_lu_x + (x_tile_min + x) * 50
                        y_geo = tile_lu_y - (y_tile_min + y) * 50

                        depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, \
                        mid_h, rim_h, rim_w = get_terr_factors(dem_crater_area, dem_crater_area_255, model_path, num_profiles=8)
                        c2.append([x_geo, y_geo, diam, depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w, id, file.split('.')[0]])
                        ids.append(id)
                        # crater_seg_morf.append(morf)
                        # crater_seg_sjb.append(sjb)
                        id += 1
                    #寫入表格
                else:
                    # print("x_geo, y_geo:", x_geo, y_geo)
                    c1.append([tile_lu_x+x_geo*50, tile_lu_y-y_geo*50, diam, id, file.split('.')[0]])
                id += 1

    # crater_data = {
    #     'ID': ids,
    #     'Morf': crater_seg_morf,
    #     'Depth/Diameter Ratio': crater_seg_sjb,
    #     'geometry': crater_seg_polys  # 这里将多边形列表添加到数据字典中
    # }
    # gdf = gpd.GeoDataFrame(crater_data)
    # gdf.crs = get_crs(tiff_ref_path)
    # gdf.to_file(shp_save_path+"2.shp", driver='ESRI Shapefile')
    # print("pred crater_num", pred_num)
    return c1, c2

if __name__ == "__main__":
    #地址
    #result_json_path = '/home/xgq/Desktop/HF/yunshi/results/ms2craternet.json'
    result_json_path = '/home/xgq/Desktop/HF/yunshi/results/exp_image_data.json'
    net_model_path = '/home/xgq/Desktop/HF/yunshi/ultralytics-m/ultralytics-modify/runs/detect/bi_3sff_dcn/train113/weights/best.pt'

    #ms2craternet region
    # rgb_test_path = '/home/xgq/Desktop/HF/yunshi/data/coco_rgb/val2017/'
    # dem_test_path = '/home/xgq/Desktop/HF/yunshi/data/coco_dem/alldata/'
    # dem_tiles_path = '/home/xgq/Desktop/HF/yunshi/data/test_files_dem/'
    # dom_tiles_path = '/home/xgq/Desktop/HF/yunshi/data/test_files/'

    #jiezeluo region
    rgb_test_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dom_clips/'
    dem_test_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dem_clips/'
    dem_tiles_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dem/'
    dom_tiles_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dom/'


    tiff_ref_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dem/HMC_13E10_dt5.tif'
    swin_model_path = '/home/xgq/Desktop/HF/yunshi/swin_transformer/weights/morf_best_model-2.pth'
    rf_model_path = '/home/xgq/Desktop/HF/yunshi/machine_learing/weights/profile_cls_model_8.joblib'

    shp_save_path = '/home/xgq/Desktop/HF/yunshi/results/poly_shp/craters.shp'
    out_dir = '/home/xgq/Desktop/HF/yunshi/results/'
    csv1 = '/home/xgq/Desktop/HF/yunshi/results/smaller_3km_results_exp.csv'
    csv2 = '/home/xgq/Desktop/HF/yunshi/results/bigger_3km_results_exp.csv'
    csv3 = '/home/xgq/Desktop/HF/yunshi/results/bigger_50km_results_exp.csv'

    with open(result_json_path, 'r') as file:
        loaded_dict = json.load(file)

    c1 = [] #直径小于3km
    c2 = [] #直径大于3km
    pixel_size = (50, 50)
    c1, c2 = bbox2atts(dom_tiles_path, dem_tiles_path, loaded_dict, rf_model_path, rgb_test_path, dem_test_path, c1, c2, tiff_ref_path, pixel_size, shp_save_path) #这里包含提取大撞击坑地形参数的过程

    #保存c1到表格
    c_array_small_df = pd.DataFrame(c1, columns=['x', 'y', 'diam', 'id', 'file'])
    c_array_small_df.to_csv(csv1, index=False)
    # # #保存c2坑到表格
    c_array_big_df = pd.DataFrame(c2, columns=['x', 'y', 'diam', 'depth', 'sjb', 'morf', 'peak_h', 'pit_d', 'floor_h', 'floor_distance', 'floor_s_ratio', 'valley_bia', 'mid_h', 'rim_h', 'rim_w', 'id', 'file'])
    c_array_big_df.to_csv(csv2, index=False)



#SAM
# cropped_dem_img = scale_to_ubyte(cropped_dem_img_0)
# cropped_dem_img = np.stack([cropped_dem_img] * 3, axis=-1)
#image = cv2.cvtColor(cropped_dem_img, cv2.COLOR_BGR2RGB)
# s = (diam/2)**2*3.14
                    # SAM
                    # masks = segment_crater(image)
                    # # 提取所有内部地貌块
                    # internal_blocks_info = extract_segments(image, w, h, diam, masks, min_area_threshold=s/80,
                    #                                         buffer_size=9, threshold_flat=diam*3/4, threshold_area=[s/5, s/3],threshold_std=12)
                    # # 找到综合评价后面积最大、位置最中心的一块，作为整个图的分类
                    # internal_blocks_info.sort(key=lambda block_info: evaluate_block(block_info[0], block_info[2],
                    #                                                                 w, h), reverse=True)
                    # centroid, terrain_type, area, segment = internal_blocks_info[0]
                    # seg = np.array(segment * 255, dtype=np.uint8)
                    # seg_img = Image.fromarray(seg, mode='L')
                    # seg_img.show()
                    # print("terrain_type:", terrain_type)
                    # morf_x = centroid[0]
                    # morf_y = centroid[1]
                    #image.show()

                    #x_bia, y_bia, sjb, diff_index, morf, morf_x, morf_y, features_list = get_terr_factors(cropped_dem_img, diam)
                    #用訓練好的模型預測退化程度
                    #degration_lv = rf(diam, surface_roughness, elevation_variation_coefficient, mean_slope, mean_curvature,hp_v, vp_v, dp1_v, dp2_v, sjb)
                    #更新撞擊坑坐標
                    # x += x_bia
                    # y += y_bia
                    # morf_x = morf_x + x_min
                    # morf_y = morf_y + y_min
                    #c2.append([x, y, diam, sjb, diff_index, morf, morf_x, morf_y, id, file.split('.')[0]])