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
    seg = np.stack([seg] * 3, axis=-1)
    seg_img = Image.fromarray(seg)
    seg_img.show()

def show_segment_list(segments):
    for segment in segments:
        show_seg(segment)

# def show_segment_list(segments):
#     # 计算分割块的数量
#     num_segments = len(segments)
#
#     # 创建一个新的图像窗口
#     plt.figure(figsize=(num_segments * 3, 3))  # 根据分割块的数量调整图像窗口大小
#
#     for i, segment in enumerate(segments):
#         # 将每个分割块转换为白色前景（1 -> 255）和黑色背景（0 -> 0）
#         seg = np.array(segment * 255, dtype=np.uint8)  # 前景255，背景0
#         seg_rgb = np.stack([seg] * 3, axis=-1)  # 将单通道转换为三通道
#
#         # 在图像窗口中绘制每个分割块
#         plt.subplot(1, num_segments, i + 1)  # 1行num_segments列
#         plt.imshow(seg_rgb)
#         plt.axis('off')  # 不显示坐标轴
#
#     # 展示所有分割块
#     plt.tight_layout()
#     plt.show()

def segment_crater(img):
    sam_checkpoint = "/home/xgq/Desktop/HF/yunshi/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    # plt.figure(figsize=(20, 20))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    # 初始化SAM模型
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # 生成掩码
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)

    # 将所有掩码合并成一个大掩码
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask[mask['segmentation']] = 1

    # 查找未被覆盖的区域
    unmasked_area = np.where(combined_mask == 0, 1, 0).astype(np.uint8)

    if unmasked_area.any():
        # 计算未覆盖区域的属性
        num_labels, labels_im = cv2.connectedComponents(unmasked_area)

        for label in range(1, num_labels):
            # 创建掩码
            area_mask = (labels_im == label).astype(np.uint8)

            # 计算面积
            area = np.sum(area_mask)

            # 计算质心
            M = cv2.moments(area_mask)
            if M['m00'] != 0:
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])
            else:
                centroid_x, centroid_y = 0, 0  # 避免除以零的情况

            # 添加新的掩码及其属性
            masks.append({
                'segmentation': area_mask,
                'area': area,
                'centroid': (centroid_x, centroid_y)
            })

    return masks

def extract_segments(w, h, sorted_masks, min_area_threshold):
    segments = []
    centroids = []
    areas = []

    for mask in sorted_masks:
        segmentation = mask['segmentation']
        #可视化
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # image[segmentation] = color_mask
        # segment = img * np.expand_dims(segmentation, axis=-1)
        segments.append(segmentation)
        centroid = center_of_mass(segmentation)
        centroids.append(centroid)
        area = mask['area']
        areas.append(area)

    # show_segment_list(segments)

    # 过滤掉面积小于最小阈值的分割块、质心偏离的分割块
    filtered_segments_or = []
    filtered_centroids = []
    filtered_areas = []
    outside_segments = []
    noise_segments = []

    # print("w, h:", w, h)

    for segment, centroid, area in zip(segments, centroids, areas):
        # print("centroid:", centroid)
        if area >= min_area_threshold:
            #if 0.2 * h <= centroid[0] <= 0.8 * h and 0.2 * w <= centroid[1] <= 0.8 * w and is_touching_edge(segment)==False:
            if 0.2 * h <= centroid[0] <= 0.8 * h and 0.2 * w <= centroid[1] <= 0.8 * w \
                    and touching_edge_nums(segment)<=3:

                filtered_segments_or.append(segment)
                # print("len(filtered_segments_or):", len(filtered_segments_or))
                filtered_centroids.append(centroid)
                filtered_areas.append(area)
            else:
                outside_segments.append(segment)
        else:
            noise_segments.append(segment)

    # if len(filtered_segments_or) == 0 and len(outside_segments) > 0 :
    #     #从outside_segments中筛选包围框长宽比最接近1的, 加入filtered_segments_or
    #     target_aspect_ratio = 1.0
    #     aspect_ratios = [bounding_box_aspect_ratio(seg) for seg in outside_segments]
    #     index_closest = closest_aspect_ratio(target_aspect_ratio, aspect_ratios)
    #     best_segment = outside_segments[index_closest]
    #     if touching_edge_nums(best_segment)<=3:
    #         filtered_segments_or.append(best_segment)
            # show_segment_list(filtered_segments_or)
    # show_segment_list(filtered_segments_or)
    # show_bg_segments(outside_segments)
    # print("len(outside_segments):", len(outside_segments))
    # print("len(noise_segments):", len(noise_segments))
    # for seg in outside_segments:
    #     show_seg(seg)

    # 处理块之间的重叠情况
    filtered_segments = resolve_overlaps(filtered_segments_or)
    # print("len(filtered_segments):", len(filtered_segments))
    # show_different(filtered_segments, filtered_segments_or)
    if len(filtered_segments)==0:
        print("no filtered segments!")
    # show_segment_list(filtered_segments)

    # 查找坑壁块(带有空洞的分割块)
    wall_segments = []
    wall_centroids = []
    wall_areas = []
    remaining_segments = []
    remaining_centroids = []
    remaining_areas = []
    for segment, centroid, area in zip(filtered_segments, filtered_centroids, filtered_areas):
        # has_large_holes补上小空洞，保留大空洞，将具有大空洞的块作为坑壁块
        if has_large_holes(segment, min_area_threshold//5) and area >= min_area_threshold:
            wall_segments.append(segment)
            wall_centroids.append(centroid)
            wall_areas.append(area)
        else:
            remaining_segments.append(segment)
            remaining_centroids.append(centroid)
            remaining_areas.append(area)
            # 分类剩余的内部地貌块
    # show_segment_list(remaining_segments)

    internal_blocks_info = []
    for segment, centroid, area in zip(remaining_segments, remaining_centroids, remaining_areas):
        # 排除质心靠近图像边缘的块
        # 对每个实心块做分类？  还是改为，只对最中心的块做分类？
        # terrain_type = classify_terrain(segment,  area, img, buffer_size, threshold_flat, threshold_area, threshold_std)
        # info = (centroid, terrain_type, area, segment)
        #terrain_type = classify_terrain(segment, area, img, buffer_size, threshold_flat, threshold_area, threshold_std)
        info = (centroid, area, segment)
        internal_blocks_info.append(info)

    crater_wall_info = []
    for segment, centroid, area in zip(wall_segments, wall_centroids, wall_areas):
        info = (centroid, area, segment)
        crater_wall_info.append(info)

    return internal_blocks_info, crater_wall_info, wall_segments

# def has_large_holes(binary_image, hole_area_threshold):
#         # [0, 0, 0, 0, 0]
#         # [0, 1, 1, 1, 0]
#         # [0, 1, 0, 1, 0] # 这里有一个空洞
#         # [0, 1, 1, 1, 0]
#         # [0, 0, 0, 0, 0]
#         # 计算binary_image质心
#     # 可视化
#     im1 = Image.fromarray(binary_image)
#     im1.show()
#     # 填充空洞
#     filled_image = binary_fill_holes(binary_image)
#     # 找到填充的部分（即空洞）进行逻辑异或操作。
#     # 逻辑异或操作的结果是在两个输入中恰好有一个为True（在二值图像中对应1）时返回True，
#     # 否则返回False（在二值图像中对应0）。
#     holes = np.logical_xor(binary_image, filled_image)
#         # [0, 0, 0, 0, 0],
#         # [0, 0, 0, 0, 0],
#         # [0, 0, 1, 0, 0],  # 只有这里是1，表示原来的空洞位置
#         # [0, 0, 0, 0, 0],
#         # [0, 0, 0, 0, 0]
#     # 使用label函数标记不同的空洞区域，并获取它们的面积
#     labels, num_features = label(holes)
#     sizes = np.bincount(labels.ravel())[1:]  # 排除标签为0的背景
#     # 检查是否有大于给定阈值的空洞面积
#     has_large_hole = np.any(sizes > hole_area_threshold)
#     return int(has_large_hole)  # 返回1表示有大于阈值的空洞，返回0表示没有

def has_large_holes(binary_image, hole_area_threshold):
    # 可视化
    # im1 = Image.fromarray((binary_image * 255).astype(np.uint8))
    # im1.show()

    # 填充所有空洞
    filled_image = binary_fill_holes(binary_image)

    # 找到填充的部分（即空洞）
    holes = np.logical_xor(binary_image, filled_image)

    # 使用label函数标记不同的空洞区域，并获取它们的面积
    lbls, num_features = label(holes)
    sizes = np.bincount(lbls.ravel())[1:]  # 排除标签为0的背景

    # 检查是否有大于给定阈值的空洞面积
    has_large_hole = np.any(sizes > hole_area_threshold)

    # if not has_large_hole:
    #     # 如果没有大空洞，使用最邻近插值法填补小空洞
    #     for lbl in range(1, num_features):
    #         # 找到当前空洞区域的点
    #         hole_points = np.where(lbls == lbl)
    #         for point in zip(hole_points[0], hole_points[1]):
    #             # 获取当前点的3x3邻域
    #             row_min = max(0, point[0] - 1)
    #             row_max = min(binary_image.shape[0], point[0] + 2)
    #             col_min = max(0, point[1] - 1)
    #             col_max = min(binary_image.shape[1], point[1] + 2)
    #             neighbors = binary_image[row_min:row_max, col_min:col_max]
    #
    #             # 找到邻域中最邻近的非零点
    #             neighbor_coords = np.argwhere(neighbors)
    #             if len(neighbor_coords) > 0:
    #                 # 选择最近的点，这里我们选择邻域中最上面的点
    #                 nearest_neighbor = neighbor_coords[0]
    #                 # 将原空洞点设置为最近邻域点的值
    #                 binary_image[point[0], point[1]] = binary_image[
    #                     nearest_neighbor[0] + row_min - 1,
    #                     nearest_neighbor[1] + col_min - 1
    #                 ]

    # 可视化最终的二值图像
    # plt.imshow(binary_image, cmap='gray')
    # plt.title('Final Binary Image')
    # plt.axis('off')
    # plt.show()

    # 返回是否有大于阈值的空洞
    return has_large_hole

def classify_terrain(segment, area, img, buffer_size, threshold_flat, threshold_area, threshold_std):
    # segment: 地貌区域的掩膜
    # centroid: 地貌的中心点坐标（但在这个函数中可能不需要使用）
    # area: 地貌的面积（可能不需要直接用于分类）
    # img: 包含高度信息的图像
    # buffer_size: 缓冲区的大小（例如，膨胀操作的核大小-1）
    # print(segment.shape) #(246, 287, 3)
    # print(img.shape) #(246, 287, 3)
    # 提取地貌区域的高度信息
    terrain_heights = img[segment]
    # 计算地貌区域的平均高度和标准差
    mean_height = np.mean(terrain_heights)
    std_height = np.std(terrain_heights)
    # height_range = np.ptp(terrain_heights)
    # 创建缓冲区掩膜
    buffered_segment = create_buffer(segment, buffer_size)  # 加1是因为核大小是奇数
    # 提取缓冲区的高度信息（只包括地貌区域外部的像素）
    buffer_heights = img[buffered_segment & ~segment]
    # 如果缓冲区没有像素，则无法分类
    if buffer_heights.size == 0:
        return "Cannot classify due to lack of buffer data"
    # 计算缓冲区的平均高度
    mean_buffer_height = np.mean(buffer_heights)
    # 根据高度差异分类
    print("std_height:", std_height)
    if np.abs(mean_height - mean_buffer_height) > threshold_flat and area <= threshold_area[0]:  # threshold_flat是平底型的阈值
        if mean_height > mean_buffer_height:
            return "Peak"
        else:
            return "Pit"
    else:
        #if area > threshold_area[1] and std_height < threshold_std:
        if std_height < threshold_std:
            return "Flat"
        else:
            return "simple"

    # if np.abs(mean_height - mean_buffer_height) < threshold_flat:  # threshold_flat是平底型的阈值
    #     if area > threshold_area:  #segment边缘的平滑程度阈值？
    #         return "Flat"
    #     else:
    #         return "simple"
    # elif mean_height > mean_buffer_height:  # 内高外低
    #     return "Peak"
    # else:  # 内低外高
    #     return "pit"

def create_buffer(segment, kernel_size):
    # 使用形态学膨胀来模拟缓冲区
    # kernel_size是膨胀操作使用的核的大小（奇数），它决定了缓冲区的宽度
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    buffered_segment = cv2.dilate(segment.astype(np.uint8), kernel, iterations=1)
    return buffered_segment

def evaluate_block(centroid, area, w, h):
    # 假设图像宽度和高度相等，简化为 w = h
    # 你可以根据实际需要调整这部分来更好地适应你的评价标准
    center_distance = min(centroid[0], w - centroid[0]) + min(centroid[1], h - centroid[1]) #质心到图像四个边界的最小距离之和。这个值越小，说明质心越靠近图像的中心

    # 面积越大、离中心越近则评价分数越高（负号表示我们希望得到更大的分数）
    score = area/(w*h) - center_distance / (w + h)  # 根据需要调整分数计算的细节
    return score

