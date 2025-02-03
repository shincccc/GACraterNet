import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

def get_profile(image, num_profile):
    rows, cols = image.shape
    center_x, center_y = cols // 2, rows // 2
    angles = np.linspace(0, np.pi, num_profile, endpoint=False)
    profiles = []

    for angle in angles:
        p1, p2 = find_boundary_points(center_x, center_y, cols, rows, angle)
        if p1 is None or p2 is None:
            profiles.append([])
            continue

        x0, y0 = p1  # 输入格式：(col, row)
        x1, y1 = p2
        cc, rr = bresenham_line(x0, y0, x1, y1)  # 返回 (cols, rows)

        profile = []
        for r, c in zip(rr, cc):
            if 0 <= r < rows and 0 <= c < cols:
                profile.append(image[r, c])
            else:
                profile.append(0)
        profiles.append(remove_noise(profile))

    return profiles

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return np.array(points).T  # 返回 (cols, rows)

def find_boundary_points(cx, cy, cols, rows, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    points = []
    if cos_theta != 0:
        # 左边界 (x=0)
        t_left = (0 - cx) / cos_theta
        y_left = cy + t_left * sin_theta
        if 0 <= y_left < rows:
            points.append((t_left, (0, int(y_left))))
        # 右边界 (x=cols-1)
        t_right = (cols - 1 - cx) / cos_theta
        y_right = cy + t_right * sin_theta
        if 0 <= y_right < rows:
            points.append((t_right, (cols - 1, int(y_right))))

    if sin_theta != 0:
        # 上边界 (y=0)
        t_top = (0 - cy) / sin_theta
        x_top = cx + t_top * cos_theta
        if 0 <= x_top < cols:
            points.append((t_top, (int(x_top), 0)))
        # 下边界 (y=rows-1)
        t_bottom = (rows - 1 - cy) / sin_theta
        x_bottom = cx + t_bottom * cos_theta
        if 0 <= x_bottom < cols:
            points.append((t_bottom, (int(x_bottom), rows - 1)))
            
    if not points:
        return None, None
    points.sort(key=lambda x: x[0])
    p_min = points[0][1]
    p_max = points[-1][1]
    return p_min, p_max

def remove_noise(profile, window_size=3):
    window_size = len(profile) // 20
    if window_size < 1:
        window_size = 3
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_profile = np.convolve(profile, weights, 'valid')  
    return smoothed_profile

def show_profiles(profile_list):
    num_profiles = len(profile_list)

    num_rows = (num_profiles + 1) // 2  # 计算子图行数

    plt.figure(figsize=(12, 8))

    for i in range(num_profiles):
        plt.subplot(num_rows, 2, i + 1)
        plt.plot(range(len(profile_list[i])), profile_list[i], color='C{}'.format(i))  # 使用不同的颜色
        plt.title('Profile {}'.format(i + 1))
        plt.xlabel('Pixel')
        plt.ylabel('Elevation')

    plt.tight_layout()
    plt.show()

def show_profiles_index(profile_list):
    num_profiles = len(profile_list)
    num_rows = (num_profiles + 1) // 2  # Calculate the number of rows for subplots

    plt.figure(figsize=(12, 8))

    for i in range(num_profiles):

        i1, i2, i3, i4 = get_structure_boundaries(profile_list[i], len(profile_list[i]))

        plt.subplot(num_rows, 2, i + 1)
        plt.plot(range(len(profile_list[i])), profile_list[i], color='C{}'.format(i))  # Plot the profile
        # plt.title('Profile {}'.format(i + 1))
        # plt.xlabel('Pixel')
        # plt.ylabel('Elevation')

        # Plot the four index
        indexes = [i1, i2, i3, i4]
        for idx in indexes:
            if idx < len(profile_list[i]):  # Make sure the index is within bounds
                plt.scatter(idx, profile_list[i][idx], color='red')  # Mark the point with a red dot
                plt.text(idx, profile_list[i][idx], f'({idx},{profile_list[i][idx]:.2f})', color='black', fontsize=11, ha='left')

    plt.tight_layout()
    plt.show()

def get_indexs(fh, length, avg_elevation):
    fh = np.array(fh)  # 确保fh是NumPy数组
    mean_h = avg_elevation
    l = len(fh)

    max_indices = []
    for i in range(2, l - 1):
        if fh[i] >= fh[i - 1] and fh[i] > fh[i + 1]:
            max_indices.append(i)

    max_indices_unique = np.unique(max_indices)
    index1 = 0
    index4 = length - 1

    if len(max_indices_unique) > 0:
        best_idx_l = None
        best_idx_r = None
        for idx in max_indices_unique:
            if fh[idx] > fh[0] and idx < l // 4 and fh[idx] > avg_elevation:
                if best_idx_l is None or fh[idx] > fh[best_idx_l]:
                    best_idx_l = idx
                if best_idx_l is not None:
                    index1 = best_idx_l

            if fh[idx] > fh[-1] and idx > 3*l // 4 and fh[idx] > avg_elevation:
                if best_idx_r is None or fh[idx] > fh[best_idx_r]:
                    best_idx_r = idx
                if best_idx_r is not None:
                    index4 = best_idx_r

    min_indices = []
    for i in range(2, l - 1):
        if fh[i] < fh[i - 1] and fh[i] <= fh[i + 1]:
            min_indices.append(i)

    min_indices_unique = np.unique(min_indices)
    index2 = length // 2 -1
    index3 = length // 2 -1
    best_score1 = 0
    best_score2 = 0

    for id in min_indices_unique:
        if index1 < id < l // 2:
        #if fh[id] <= mean_h and index1 < id < l//2:
            sl1 = fh[max(0, id - (l // 4))] - fh[id]
            # sl2 = fh[min(id + (l // 10), l - 1)] - fh[id]
            if sl1 > best_score1:
                best_score1 = sl1
                index2 = id-1
        if l // 2 < id < index4:
        #if fh[id] <= mean_h and l//2 < id < index4:
            sr1 = fh[min(id + (l // 4), l - 1)] - fh[id]
            # sr2 = fh[max(0, id - (l // 10))] - fh[id]
            if sr1 > best_score2:
                best_score2 = sr1
                index3 = id+1
                
    return index1, index2, index3, index4

def get_structure_boundaries(horizontal_profile, length):
    #smooth_window = len(horizontal_profile) // 10
    # horizontal_profile = smooth_profile(horizontal_profile, smooth_window)
    avg_elevation = np.mean(horizontal_profile)
    index1, index2, index3, index4 = get_indexs(horizontal_profile, length, avg_elevation)
    return index1, index2, index3, index4


if __name__ == "__main__":

    #img = '/home/xgq/Desktop/HF/yunshi/machine_learing/dataset_eh/filltered_CpxFF/06-0-001522.jpg'
    img = '/home/xgq/Desktop/HF/yunshi/machine_learing/cropped_train_dem/special/06-0-001752.jpg'
    img_array = np.array(Image.open(img))
    p_list = get_profile(img_array, num_profile=4)
    # show_profiles(p_list)
    show_profiles_index(p_list)
