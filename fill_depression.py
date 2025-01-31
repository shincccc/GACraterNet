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
    # min_val0 = np.min(array)
    # max_val0 = np.max(array)
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

def extract_region(dem_region, d, swin_model_path, id, file_name):
    dem_region_uint8 = scale_to_ubyte(dem_region)
    dem_region_uint8 = np.stack([dem_region_uint8] * 3, axis=-1)
    # max_h = np.max(dem_region_uint8)
    # min_h = np.min(dem_region_uint8)
    # sjb = (max_h - min_h) / (d * 50)
    img = Image.fromarray(dem_region_uint8)
    img.save("/home/xgq/Desktop/HF/yunshi/results/fld_results/" + file_name +'_fld_'+ str(id) + '.jpg')
    #morf, peak_h, pit_d, floor_distance, rim_h, rim_w = get_terr_factors(dem_region_uint8, rgb_array = [], model_path=model_path, num_profiles = 12)
    depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w = get_terr_factors(
        dem_region, dem_region_uint8, swin_model_path, num_profiles=12)

    return depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w

def calculate_iou(bbox1, bbox2): #为了去除重复框
    """计算两个边界框的交并比（IoU）"""
    x1 = max(bbox1[1], bbox2[1])
    y1 = max(bbox1[0], bbox2[0])
    x2 = min(bbox1[3], bbox2[3])
    y2 = min(bbox1[2], bbox2[2])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    bbox1_area = (bbox1[3] - bbox1[1] + 1) * (bbox1[2] - bbox1[0] + 1)
    bbox2_area = (bbox2[3] - bbox2[1] + 1) * (bbox2[2] - bbox2[0] + 1)

    iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
    assert iou >= 0
    assert iou <= 1.0
    return iou

#获取连续区域
# def get_best_region_bbox(dataset, area_threshold, density_threshold, value_threshold, iou_threshold=0.3):
#     # 获取影像数据
#     b_image = dataset.ReadAsArray().astype(np.uint8)
#     # 转到0-255
#     # b_image = scale_to_ubyte(b_image)
#     # 使用OTSU阈值处理图像，将非零值设置为1
#     _, binary_image = cv2.threshold(b_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # 清除边界噪声
#     binary_image = clear_border(binary_image)
#     # 标记连通区域
#     labeled_array, num_features = label(binary_image)
#     # 获取每个区域的边界框和属性信息
#     regions = regionprops(labeled_array)
#     # 存储有效的区域边界框
#     valid_objects = []
#     # 迭代每个区域
#     for region in regions:
#         density = region.area / (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
#         region_mean_value = np.mean(b_image[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]])
#         # 如果当前区域的面积大于等于阈值，则可能是有效的
#         if region.area >= area_threshold and density > density_threshold and region_mean_value > value_threshold:
#             valid_objects.append((region.bbox, region.area))  # 存储bbox和面积以便后续比较
#     # 去除重叠的bbox，并基于面积选择最佳的一个
#     best_bboxes = []
#     used_labels = set()
#     for bbox, area in sorted(valid_objects, key=lambda x: x[1], reverse=True):  # 按面积降序排序
#         keep = True
#         for prev_bbox in best_bboxes:
#             iou = calculate_iou(bbox, prev_bbox)
#             if iou >= iou_threshold:  # 如果重叠度超过阈值，则不保留这个bbox
#                 keep = False
#                 break
#         if keep and bbox[0] not in used_labels:  # 确保标签没有被使用过（防止bug）
#             best_bboxes.append(bbox)
#             used_labels.add(bbox[0])  # bbox的第一个元素是y的最小值，这里假设它是唯一的标签
#     # 更新有效的区域列表
#     objects = best_bboxes
#     return objects, b_image, labeled_array
def calculate_iou(bbox1, bbox2):
    # 计算两个边界框的IOU (Intersection over Union)
    x1, y1, x2, y2 = bbox1
    x1b, y1b, x2b, y2b = bbox2

    # 计算交集区域
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    # 确保交集区域存在
    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        intersection_area = 0

    # 计算联合区域
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2b - x1b) * (y2b - y1b)
    union_area = bbox1_area + bbox2_area - intersection_area

    # 计算IOU
    return intersection_area / union_area

def get_best_region_bbox(dataset, area_threshold, density_threshold, iou_threshold=0.3):
    # 获取影像数据
    b_image = dataset.ReadAsArray().astype(np.uint16)

    # 可视化异常值去除后影像
    o_image = scale_to_ubyte(b_image)
    plt.figure(figsize=(6, 6))
    plt.imshow(b_image, cmap='gray')
    plt.title("o Image")
    plt.axis('off')
    plt.show()

    #异常值去除
    b_image[b_image == 32768] = 0
    min_val0 = np.min(b_image)
    max_val0 = np.max(b_image)
    print("min_val0, max_val0:", min_val0, max_val0)

    # 可视化异常值去除后影像
    b_image = scale_to_ubyte(b_image)
    plt.figure(figsize=(6, 6))
    plt.imshow(b_image, cmap='gray')
    plt.title("Blured Image")
    plt.axis('off')
    plt.show()

    # 使用OTSU阈值处理图像，将非零值设置为1
    #_, binary_image = cv2.threshold(b_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_image = cv2.threshold(b_image, 0, 1, cv2.THRESH_BINARY)


    # 可视化二值化图像
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary Image (OTSU Thresholding)")
    plt.axis('off')
    plt.show()

    # 清除边界噪声
    binary_image = clear_border(binary_image)

    # 可视化清除边界噪声后的图像
    # plt.figure(figsize=(6, 6))
    # plt.imshow(binary_image, cmap='gray')
    # plt.title("Binary Image after Clear Border")
    # plt.axis('off')
    # plt.show()

    # 标记连通区域
    labeled_array, num_features = label(binary_image)
#---------------------------------------------
    # 计算每个连通区域的面积并过滤
    area_threshold = 60  # 设置面积阈值，根据需要调整
    filtered_labeled_array = np.zeros_like(labeled_array)  # 创建一个空数组用于存储过滤后的结果

    # 遍历每个连通区域
    for region in regionprops(labeled_array):
        if region.area >= area_threshold:  # 如果区域面积大于阈值
            # 保留该区域
            filtered_labeled_array[labeled_array == region.label] = region.label
#-------------------------------------------------
    # 可视化标记的连通区域
    plt.figure(figsize=(6, 6))
    #plt.imshow(labeled_array, cmap='nipy_spectral')
    plt.imshow(labeled_array, cmap='gray')
    plt.title("Labeled Regions")
    plt.axis('off')
    plt.show()

    # 获取每个区域的边界框和属性信息
    regions = regionprops(labeled_array)

    # 存储有效的区域边界框
    valid_objects = []

    # 迭代每个区域
    for region in regions:
        density = region.area / (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
        #region_mean_value = np.mean(b_image[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]])

        # 如果当前区域的面积大于等于阈值，则可能是有效的
        #if region.area >= area_threshold and density > density_threshold and region_mean_value > value_threshold:
        if region.area >= area_threshold and density > density_threshold:
            valid_objects.append((region.bbox, region.area))  # 存储bbox和面积以便后续比较

    # 可视化有效区域的边界框
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(b_image, cmap='gray')
    for bbox, _ in valid_objects:
        minr, minc, maxr, maxc = bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.title("Valid Region Bboxes")
    plt.axis('off')
    plt.show()

    # 去除重叠的bbox，并基于面积选择最佳的一个 (在多个边界框检测到同一目标时，选择最佳的一个边界框,相当于NMS)。
    best_bboxes = []
    used_labels = set()
    for bbox, area in sorted(valid_objects, key=lambda x: x[1], reverse=True):  # 按面积降序排序
        keep = True
        for prev_bbox in best_bboxes:
            iou = calculate_iou(bbox, prev_bbox)
            if iou >= iou_threshold:  # 如果重叠度超过阈值，则不保留这个bbox
                keep = False
                break
        if keep and bbox[0] not in used_labels:  # 确保标签没有被使用过（防止bug）
            best_bboxes.append(bbox)
            used_labels.add(bbox[0])  # bbox的第一个元素是y的最小值，这里假设它是唯一的标签

    # 可视化最终的最佳边界框
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(b_image, cmap='gray')
    for bbox in best_bboxes:
        minr, minc, maxr, maxc = bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
    plt.title("Best Region Bboxes after IOU Filtering")
    plt.axis('off')
    plt.show()

    # 返回有效的区域边界框
    objects = best_bboxes
    return objects, b_image, labeled_array

def get_region_bbox(c3, gcc_tiff_path, dem_tiff_path, swin_model_path, rf_model_path, file_name, area_threshold=40000, density_threshold=0.1):
    dataset = gdal.Open(gcc_tiff_path)

    print(gcc_tiff_path)
    geotransform = dataset.GetGeoTransform()
    # 获取地理坐标信息
    origin_x = geotransform[0]
    origin_y = geotransform[3]

    dem_dataset = gdal.Open(dem_tiff_path).ReadAsArray()
    objects, b_image, labeled_array = get_best_region_bbox(dataset, area_threshold, density_threshold)
    #处理objects可能含有重复的对象

    for i, obj in enumerate(objects):

        x_bbox = obj[1]
        y_bbox = obj[0]
        w = abs(obj[3] - obj[1])
        h = abs(obj[2] - obj[0])
        d = int((w+h)//2*1.2)
        xc = w//2
        yc = h//2

        #if is_approx_circle and r >= 60 and abs(w / h) >= 0.70:
        if d >= 60 and abs(w/h) >= 0.40:  # 只保留 is_approx_circle 为 True 的行
            dem_region = dem_dataset[obj[0] - int(d * 0.2):obj[2] + int(d * 0.2), obj[1] - int(d * 0.2):obj[3] + int(d * 0.2)]

            if dem_region is not None and dem_region.shape[0]!=0 and dem_region.shape[1]!=0:
                depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, \
                mid_h, rim_h, rim_w = extract_region(dem_region, d, swin_model_path, i, file_name)
                c3.append([int(origin_x + (x_bbox +  xc) * 50), int(origin_y - (y_bbox +  yc) * 50), d, depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, \
                mid_h, rim_h, rim_w, str(i)+'_fld', file_name])
    return c3

if __name__=="__main__":
    gcc_path = '/home/xgq/Desktop/HF/yunshi/data/new_test_set_dem/衍生图/'
    dem_path = '/home/xgq/Desktop/HF/yunshi/data/experiment/dem/'

    gcc_path = '/home/xgq/Desktop/HF/yunshi/data/gcc1/'
    dem_path = '/home/xgq/Desktop/HF/yunshi/data/dem1/'


    # gcc_path = '/home/xgq/Desktop/HF/yunshi/data/test_files_gcc/'
    # dem_path = '/home/xgq/Desktop/HF/yunshi/data/test_files_dem/'
    rf_model_path = '/home/xgq/Desktop/HF/yunshi/machine_learing/weights/profile_cls_model_6.joblib'
    swin_model_path = '/home/xgq/Desktop/HF/yunshi/swin_transformer/weights/morf_best_model-2.pth'
    csv3 = '/home/xgq/Desktop/HF/yunshi/results/bigger_50km_results_exp.csv'

    c3 = []

    for file in os.listdir(dem_path):
        file_name = file.split('.tif')[0]
        dem_tiff_path = dem_path + file
        gcc_tiff_path = gcc_path + file_name + '_gcc.tif'
        c3 = get_region_bbox(c3, gcc_tiff_path, dem_tiff_path, swin_model_path, rf_model_path, file_name, area_threshold=40000, density_threshold=0.3)

    c_array_big_df = pd.DataFrame(c3, columns=['x', 'y', 'diam', 'depth', 'sjb', 'morf', 'peak_h', 'pit_d', 'floor_h', 'floor_distance', 'floor_s_ratio', 'valley_bia', 'mid_h', 'rim_h', 'rim_w', 'id', 'file'])
    c_array_big_df.to_csv(csv3, index=False)
