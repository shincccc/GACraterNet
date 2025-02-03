import numpy as np
import os
from get_profiles import get_profile, get_indexs, get_structure_boundaries
from machine_learing.classification_rf import predict_profile_morf, get_ml_features
from swin_transformer.predict import predict_crater
import statistics
from scipy.ndimage import zoom

def scale_to_ubyte(array):
    array_float = array.astype(np.float32)
    min_val = np.min(array_float)
    max_val = np.max(array_float)
    scaled_array = ((array_float - min_val) / (max_val - min_val)) * 255
    scaled_array = np.floor(scaled_array).astype(np.uint8)
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
    depth = np.min(dem_array)
    original_shape = dem_img2.shape
    new_shape = [256, 256, 3]
    zoom_factors = (new_shape[0] / original_shape[0], new_shape[1] / original_shape[1], 1)
    dem_img2 = zoom(dem_img2, zoom_factors, order=3)
    morf = predict_crater(dem_img2, model_path)
    # print(morf)
    depth, sjb, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, \
    mid_h, rim_h, rim_w = get_profile_features(profile_list)
    return depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w

def get_terr_factors(dem_array, dem_img2, model_path, num_profiles):
    #profile_list = get_profile(d_array, num_profiles)
    profile_list = get_profile(dem_array, num_profiles)
    # profile_list_cls = get_profile(scale_to_ubyte(dem_array), num_profiles)

    depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w = center_position_morf(profile_list, dem_array, dem_img2, model_path, num_profiles)

    return depth, sjb, morf, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w
    #return surface_roughness, elevation_variation_coefficient, mean_slope, mean_curvature, hp_v, vp_v, dp1_v, dp2_v, sjb
