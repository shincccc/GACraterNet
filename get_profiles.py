import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

# def get_profile_or(image):
#     # image = cv2.resize(image, (256, 256))
#     image_height, image_width = image.shape[:2]
#     center_pixel = (image_width // 2, image_height // 2)
#
#     # 提取水平地形剖線
#     horizontal_profile = remove_noise(image[center_pixel[1], :])
#
#     # 提取垂直地形剖線
#     vertical_profile = remove_noise(image[:, center_pixel[0]])
#
#     diagonal_profile1 = []
#     diagonal_profile2 = []
#
#     # 計算45度斜線的起始點和終點像素位置
#     diagonal_length = min(center_pixel[0], center_pixel[1])
#     start_point1 = (center_pixel[0] - diagonal_length, center_pixel[1] - diagonal_length)
#     end_point1 = (center_pixel[0] + diagonal_length, center_pixel[1] + diagonal_length)
#     start_point2 = (center_pixel[0] + diagonal_length, center_pixel[1] - diagonal_length)
#     end_point2 = (center_pixel[0] - diagonal_length, center_pixel[1] + diagonal_length)
#
#     for i in range(diagonal_length * 2):
#         x1 = start_point1[0] + i
#         y1 = start_point1[1] + i
#         x2 = start_point2[0] - i
#         y2 = start_point2[1] + i
#
#         # 確保座標在圖像範圍內
#         if (x1 < 0 or y1 < 0 or x1 >= image_width or y1 >= image_height or
#                 x2 < 0 or y2 < 0 or x2 >= image_width or y2 >= image_height):
#             continue
#
#         diagonal_profile1.append(image[y1, x1])
#         diagonal_profile2.append(image[y2, x2])
#
#     diagonal_profile1 = remove_noise(diagonal_profile1)
#     diagonal_profile1 = remove_noise(diagonal_profile1)
#
#     #fit profiles
#     # 水平地形剖線
#     return [horizontal_profile, vertical_profile, diagonal_profile1, diagonal_profile2]

# def get_profile(image, num_profile):
#     """
#     提取通过图像中心的多个径向剖线。
#
#     :param image: 2D numpy array，输入的图像数组
#     :param num_profiles: 需要提取的径向剖线数量
#     :return: 一个字典，包含每个角度的剖线数据
#     """
#
#     # 确定图像的几何中心
#     rows, cols = image.shape
#     center_x, center_y = cols // 2, rows // 2
#
#     # 角度间隔
#     angles = np.linspace(0, np.pi, num_profile, endpoint=False)
#
#     profiles = []
#
#     for angle in angles:
#         profile = []
#         for t in range(-int(max(rows, cols)/2), int(max(rows, cols)/2)):
#             x = int(center_x + t * np.cos(angle))
#             y = int(center_y + t * np.sin(angle))
#             # 确保索引在图像范围内
#             if 0 <= x < cols and 0 <= y < rows:
#                 profile.append(image[y, x])
#             else:
#                 profile.append(0)  # 超出图像边界的值设为0
#         profiles.append(remove_noise(profile))
#         #profiles.append(profile)
#
#     return profiles

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

        # 使用自定义 Bresenham 算法生成直线坐标
        x0, y0 = p1  # 输入格式：(col, row)
        x1, y1 = p2
        cc, rr = bresenham_line(x0, y0, x1, y1)  # 返回 (cols, rows)

        # 提取像素值
        profile = []
        for r, c in zip(rr, cc):
            if 0 <= r < rows and 0 <= c < cols:
                profile.append(image[r, c])
            else:
                profile.append(0)
        profiles.append(remove_noise(profile))

    return profiles

def bresenham_line(x0, y0, x1, y1):
    """Bresenham 直线算法实现"""
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
    """找到沿给定角度从中心出发与图像边界的两个交点"""
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    points = []

    # 计算与各边界的交点
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

    # 按t值排序，取最远的两点
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
    smoothed_profile = np.convolve(profile, weights, 'valid')  # 使用 'same' 来保持输入和输出的形状一致
    #简单移动平均法
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

# def get_indexs(fh, length, avg_elevation):
#     fh = np.array(fh)  # 确保fh是NumPy数组
#     mean_h = avg_elevation
#     l = len(fh)
#
#     max_indices = []
#     for i in range(2, l - 1):
#         if fh[i] >= fh[i - 1] and fh[i] > fh[i + 1]:
#             max_indices.append(i)
#
#     max_indices_unique = np.unique(max_indices)
#     index1 = 0
#     index4 = length - 1
#
#     if len(max_indices_unique) > 0:
#         best_idx_l = None
#         best_idx_r = None
#         for idx in max_indices_unique:
#             if fh[idx] > fh[0] and idx < l // 4:
#                 # 筛选出满足上一行 if 条件中的 idx 最小的点
#                 if best_idx_l is None or fh[idx] > fh[best_idx_l]:
#                     index1 = idx
#
#             if fh[idx] > fh[-1] and idx > 3*l // 4:
#                 # 筛选出满足上一行 if 条件中的 idx 最小的点
#                 if best_idx_r is None or fh[idx] > fh[best_idx_r]:
#                     index4 = idx
#
#     min_indices = []
#     for i in range(2, l - 1):
#         if fh[i] < fh[i - 1] and fh[i] <= fh[i + 1]:
#             min_indices.append(i)
#
#     min_indices_unique = np.unique(min_indices)
#     index2 = length // 2 -1
#     index3 = length // 2 -1
#     best_score1 = 0
#     best_score2 = 0
#
#     for id in min_indices_unique:
#         if fh[id] <= mean_h and 0 < id < l//2:
#             sl1 = fh[max(0, id - (l // 5))] - fh[id]
#             # sl2 = fh[min(id + (l // 10), l - 1)] - fh[id]
#             if sl1 > best_score1:
#                 best_score1 = sl1
#                 index2 = id-1
#         if fh[id] <= mean_h and l//2 < id < l:
#             sr1 = fh[min(id + (l // 5), l - 1)] - fh[id]
#             # sr2 = fh[max(0, id - (l // 10))] - fh[id]
#             if sr1 > best_score2:
#                 best_score2 = sr1
#                 index3 = id+1
#         # 计算高程波动阈值内的长度
#
#     return index1, index2, index3, index4

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
                # 筛选出满足上一行 if 条件中高程最大的点
                if best_idx_l is None or fh[idx] > fh[best_idx_l]:
                    best_idx_l = idx
                if best_idx_l is not None:
                    index1 = best_idx_l

            if fh[idx] > fh[-1] and idx > 3*l // 4 and fh[idx] > avg_elevation:
                # 筛选出满足上一行 if 条件中高程最大的点
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
        # 计算高程波动阈值内的长度

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