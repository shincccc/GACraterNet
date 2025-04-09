import numpy as np
import os
from get_profiles import get_profile, get_indexs, get_structure_boundaries
from machine_learing.classification_rf import predict_profile_morf, get_ml_features
from swin_transformer.predict import predict_crater
import statistics
from scipy.ndimage import zoom

def scale_to_ubyte(array):
    """Scale array to 0-255 range and convert to uint8."""
    array_float = array.astype(np.float32)
    min_val, max_val = np.min(array_float), np.max(array_float)
    scaled = ((array_float - min_val) / (max_val - min_val)) * 255
    return np.clip(np.floor(scaled), 0, 255).astype(np.uint8)

def get_rim(p, i1, i4):
    """Calculate rim height and width from profile points."""
    rim_h_list, rim_w_list = [], []
    l = len(p)

    if 0 < i1 <= l:
        i_min1 = np.argmin(p[0:i1])
        rim_h_list.append(p[i1] - p[i_min1])
        rim_w_list.append(np.abs(i1 - i_min1))

    if 0 < i4 < l:
        i_min2 = np.argmin(p[i4:l]) + i4
        rim_h_list.append(p[i4] - p[i_min2])
        rim_w_list.append(np.abs(i_min2 - i4))

    return (statistics.mean(rim_h_list), statistics.mean(rim_w_list)) if rim_h_list else None

def get_profile_features(p_list):
    """Extract features from profile list and compute means."""
    peak_h_list, pit_d_list, floor_distance_list = [], [], []
    rim_h_list, rim_w_list, depth_list, sjb_list = [], [], [], []
    valley_bia_list, mid_h_list = [], []

    for p in p_list:
        i1, i2, i3, i4 = get_indexs(p, len(p), np.mean(p))
        depth_list.append(np.max(p) - np.min(p))
        l, sjb, depth_rate, floor_distance, peak_h, pit_d, valley_position, floor_h, min_nums, max_nums = get_ml_features(p, i1, i2, i3, i4)
        sjb_list.append(sjb)
        peak_h_list.append(peak_h)
        pit_d_list.append(pit_d)
        floor_distance_list.append(floor_distance)
        valley_bia_list.append(abs(valley_position - 0.5))
        mid_h_list.append(depth_rate)
        rim = get_rim(p, i1, i4)
        if rim:
            rim_h, rim_w = rim
            rim_h_list.append(rim_h)
            rim_w_list.append(rim_w)

    depth, sjb = np.mean(depth_list), np.mean(sjb_list)
    peak_h, pit_d = np.mean(peak_h_list), np.mean(pit_d_list)
    floor_distance = np.mean(floor_distance_list)
    floor_s_ratio = floor_distance ** 2
    valley_bia, mid_h = np.mean(valley_bia_list), np.mean(mid_h_list)
    rim_h = round(np.mean(rim_h_list)) if rim_h_list else 0
    rim_w = round(np.mean(rim_w_list)) if rim_w_list else 0

    return depth, sjb, peak_h, pit_d, floor_h, floor_distance, floor_s_ratio, valley_bia, mid_h, rim_h, rim_w

def center_position_morf(profile_list, dem_array, dem_img2, model_path, num_profiles):
    """Predict crater morphology and compute profile features."""
    depth = np.min(dem_array)
    zoom_factors = (256 / dem_img2.shape[0], 256 / dem_img2.shape[1], 1)
    dem_img2 = zoom(dem_img2, zoom_factors, order=3)
    morf = predict_crater(dem_img2, model_path)
    return depth, *get_profile_features(profile_list)[:2], morf, *get_profile_features(profile_list)[3:]

def get_terr_factors(dem_array, dem_img2, model_path, num_profiles):
    """Generate terrain factors from DEM data using profiles and model prediction."""
    profile_list = get_profile(dem_array, num_profiles)
    return center_position_morf(profile_list, dem_array, dem_img2, model_path, num_profiles)
