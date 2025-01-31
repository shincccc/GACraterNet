import numpy as np
import os
from get_profiles import get_profile, get_indexs, get_structure_boundaries
from machine_learing.classification_rf import predict_profile_morf, get_ml_features
from swin_transformer.predict import predict_crater
import statistics
from scipy.ndimage import zoom

# def get_simp_pos_h(p_list, simp_p, h, w):
#     simp_x = []
#     simp_y = []
#     for i in simp_p:
#         p = p_list[i]
#         l = len(p)
#         i1, i2, i3, i4 = get_structure_boundaries(p, l)
#
#         if i == 0:
#             simp_x.append(i2 if i2 == i3 else np.argmin(p[i2:i3]))
#             simp_y.append(h//2 - 1)
#         if i == 1:
#             simp_x.append(w//2 - 1)
#             simp_y.append(i2 if i2 == i3 else np.argmin(p[i2:i3]))
#         if i == 2:
#             simp_x.append(i2 if i2 == i3 else np.argmin(p[i2:i3]))
#             simp_y.append(i2 if i2 == i3 else np.argmin(p[i2:i3]))
#         if i == 3:
#             simp_x.append(w-i2 if i2 == i3 else np.argmin(p[i2:i3]))
#             simp_y.append(w-i2 if i2 == i3 else np.argmin(p[i2:i3]))
#
#     simp_x = np.mean(simp_x)
#     simp_y = np.mean(simp_y)
#
#     return simp_x, simp_y


# def get_peak_pos_h(p_list, peak_p, h, w):
#     peak_x = []
#     peak_y = []
#
#     for i in peak_p:
#         p = p_list[i]
#         l = len(p)
#         i1, i2, i3, i4 = get_structure_boundaries(p, l)
#         if i == 0:
#             peak_x.append(p[i2] if i2 == i3 else np.argmax(p[i2:i3]))
#             peak_y.append(h//2 - 1)
#         if i == 1:
#             peak_x.append(w//2 - 1)
#             peak_y.append(p[i2] if i2 == i3 else np.argmax(p[i2:i3]))
#         if i == 2:
#             peak_x.append(p[i2] if i2 == i3 else np.argmax(p[i2:i3]))
#             peak_y.append(p[i2] if i2 == i3 else np.argmax(p[i2:i3]))
#         if i == 3:
#             peak_x.append(w-(p[i2] if i2 == i3 else np.argmax(p[i2:i3])))
#             peak_y.append(p[i2] if i2 == i3 else np.argmax(p[i2:i3]))
#
#     peak_x = np.mean(peak_x)
#     peak_y = np.mean(peak_y)
#
#     return peak_x, peak_y
# def get_pit_index(p, i2, i3):
#     p = p[i2:i3]
#     l = len(p)
#
#     min_indices = []
#     for i in range(2, l - 1):
#         if p[i] < p[i - 1] and p[i] <= p[i + 1]:
#             min_indices.append(i)
#     min_indices = np.unique(min_indices)
#
#     pit_h = np.inf
#     pit_index = (i2+i3)//2
#     for i in min_indices:
#         if p[i]<pit_h:
#             pit_h = p[i]
#             pit_index = i+i2
#
#     return pit_index
# def get_pit_pos_h(p_list, pit_p, h, w):
#     pit_x = []
#     pit_y = []
#
#     for i in pit_p:
#         p = p_list[i]
#         l = len(p)
#         i1, i2, i3, i4 = get_structure_boundaries(p, l)
#         if i == 0:
#             pit_x.append(get_pit_index(p, i2, i3))
#             pit_y.append(h // 2 - 1)
#         if i == 1:
#             pit_x.append(w // 2 - 1)
#             pit_y.append(get_pit_index(p, i2, i3))
#         if i == 2:
#             pit_x.append(get_pit_index(p, i2, i3))
#             pit_y.append(get_pit_index(p, i2, i3))
#         if i == 3:
#             pit_x.append(w - get_pit_index(p, i2, i3))
#             pit_y.append(get_pit_index(p, i2, i3))
#
#     pit_x = np.mean(pit_x)
#     pit_y = np.mean(pit_y)
#
#     return pit_x, pit_y
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

def get_rim(p, i1, i4):
    rim_h_list = []
    rim_w_list = []
    l = len(p)

    # Check if i1 is valid and compute first rim
    if i1 > 0 and i1 <= l:  # Ensure i1 is within valid bounds
        i_min1 = np.argmin(p[0:i1])
        h_rim1 = p[i1] - p[i_min1]
        w_rim1 = np.abs(i1 - i_min1)
        rim_h_list.append(h_rim1)
        rim_w_list.append(w_rim1)

    # Check if i4 is valid and compute second rim
    if i4 > 0 and i4 < l:  # Ensure i4 is within valid bounds
        # Adjusting to avoid empty slices
        if i4 < l:  # Ensure there's at least one element to consider
            i_min2 = np.argmin(p[i4:l]) + i4  # Correct index for min
            h_rim2 = p[i4] - p[i_min2]
            w_rim2 = np.abs(i_min2 - i4)
            rim_h_list.append(h_rim2)
            rim_w_list.append(w_rim2)

    # Return means if lists are not empty
    if rim_h_list and rim_w_list:
        return statistics.mean(rim_h_list), statistics.mean(rim_w_list)

    return None

def get_profile_features(p_list):
    peak_h_list = []
    pit_d_list = []
    floor_distance_list=[]
    rim_h_list = []
    rim_w_list = []
    depth_list = []
    sjb_list = []
    valley_bia_list = []
    mid_h_list = []

    for i, p in enumerate(p_list):
        i1, i2, i3, i4 = get_indexs(p, len(p), np.mean(p))
        depth = np.max(p) - np.min(p)
        depth_list.append(depth)
        l, sjb, depth_rate, floor_distance, peak_h, pit_d, valley_position, floor_h, min_nums, max_nums = get_ml_features(p_list[i], i1, i2, i3, i4)
        sjb_list.append(sjb)
        peak_h_list.append(peak_h)
        pit_d_list.append(pit_d)
        floor_distance_list.append(floor_distance)
        valley_bia_list.append(abs(valley_position-0.5))
        mid_h_list.append(depth_rate)
        if get_rim(p, i1, i4) is not None:
            rim_h, rim_w = get_rim(p, i1, i4)
            rim_h_list.append(rim_h)
            rim_w_list.append(rim_w)
    depth = np.mean(depth_list)
    sjb = np.mean(sjb_list)
    peak_h = np.mean(peak_h_list)
    pit_d = np.mean(pit_d_list)
    floor_distance = np.mean(floor_distance_list)
    floor_s_ratio = floor_distance ** 2
    valley_bia = np.mean(valley_bia_list)
    mid_h = np.mean(mid_h_list)
    if len(rim_h_list) > 0:
        rim_h = round(np.mean(rim_h_list))
        rim_w = round(np.mean(rim_w_list))
    else:
        rim_h = 0
        rim_w = 0
    return depth, sjb, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w

def center_position_morf(profile_list, dem_array, dem_img2, model_path, num_profiles):
    #depth = np.min(dem_array)
    #网络模型分类
    # print("dem_img2.shape", dem_img2.shape)
    # 将d_array重采样到256
    original_shape = dem_img2.shape
    new_shape = [256, 256, 3]
    zoom_factors = (new_shape[0] / original_shape[0], new_shape[1] / original_shape[1], 1)
    dem_img2 = zoom(dem_img2, zoom_factors, order=3)
    morf = predict_crater(dem_img2, model_path)
    # print(morf)
    depth, sjb, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, \
    mid_h, rim_h, rim_w = get_profile_features(profile_list)

    #剖面线随机森林分类
    # profile_dict = {'CpxCPt':[], 'CpxCPk':[], 'CpxFF':[], 'Smp':[]}
    # morf_dict = {'CpxCPt':0, 'CpxCPk':0, 'CpxFF':0, 'Smp':0}
    # for i in range(len(profile_list)):
    #     #morf = predict_profile_morf(profile_list_cls[i], model_path)
    #     # morf = predict_crater()
    #     if morf in morf_dict:
    #         morf_dict[morf]+=1
    #         profile_dict[morf].append(profile_list[i])
    #找出morf_dict中数最大的键，赋值给morf
    # if morf_dict:  # 确保字典不为空
    #     # morf = max(morf_dict, key=morf_dict.get)
    #     depth, sjb, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w = get_profile_features(profile_dict[morf])
    # else:
    #     morf = None  # 如果字典为空，设置 morf 为 None 或其他适当值
    #     depth, sjb, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w = None, None, None, None, None, None, None, None
    return depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w

def get_terr_factors(dem_array, dem_img2, model_path, num_profiles):
    #profile_list = get_profile(d_array, num_profiles)
    profile_list = get_profile(dem_array, num_profiles)
    # profile_list_cls = get_profile(scale_to_ubyte(dem_array), num_profiles)

    depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w = center_position_morf(profile_list, dem_array, dem_img2, model_path, num_profiles)

    return depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w
    #return surface_roughness, elevation_variation_coefficient, mean_slope, mean_curvature, hp_v, vp_v, dp1_v, dp2_v, sjb