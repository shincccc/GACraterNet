import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

def get_profile(image, num_profile):
    """Extract elevation profiles from image at specified angles."""
    rows, cols = image.shape
    center_x, center_y = cols // 2, rows // 2
    angles = np.linspace(0, np.pi, num_profile, endpoint=False)
    profiles = []

    for angle in angles:
        p1, p2 = find_boundary_points(center_x, center_y, cols, rows, angle)
        if p1 is None or p2 is None:
            profiles.append([])
            continue
        x0, y0, x1, y1 = *p1, *p2
        cc, rr = bresenham_line(x0, y0, x1, y1)
        profile = [image[r, c] if 0 <= r < rows and 0 <= c < cols else 0 for r, c in zip(rr, cc)]
        profiles.append(remove_noise(profile))
    return profiles

def bresenham_line(x0, y0, x1, y1):
    """Generate points along a line using Bresenham's algorithm."""
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
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
    return np.array(points).T

def find_boundary_points(cx, cy, cols, rows, angle):
    """Find intersection points of a line with image boundaries."""
    cos_theta, sin_theta = np.cos(angle), np.sin(angle)
    points = []
    if cos_theta != 0:
        t_left, t_right = (0 - cx) / cos_theta, (cols - 1 - cx) / cos_theta
        y_left, y_right = cy + t_left * sin_theta, cy + t_right * sin_theta
        if 0 <= y_left < rows:
            points.append((t_left, (0, int(y_left))))
        if 0 <= y_right < rows:
            points.append((t_right, (cols - 1, int(y_right))))
    if sin_theta != 0:
        t_top, t_bottom = (0 - cy) / sin_theta, (rows - 1 - cy) / sin_theta
        x_top, x_bottom = cx + t_top * cos_theta, cx + t_bottom * cos_theta
        if 0 <= x_top < cols:
            points.append((t_top, (int(x_top), 0)))
        if 0 <= x_bottom < cols:
            points.append((t_bottom, (int(x_bottom), rows - 1)))
    return (points[0][1], points[-1][1]) if points else (None, None)

def remove_noise(profile, window_size=3):
    """Smooth profile using a moving average filter."""
    window_size = max(len(profile) // 20, 3)
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(profile, weights, 'valid')

def show_profiles(profile_list):
    """Display profiles in a grid of subplots."""
    num_profiles, num_rows = len(profile_list), (len(profile_list) + 1) // 2
    plt.figure(figsize=(12, 8))
    for i, profile in enumerate(profile_list):
        plt.subplot(num_rows, 2, i + 1)
        plt.plot(range(len(profile)), profile, color=f'C{i}')
        plt.title(f'Profile {i + 1}')
        plt.xlabel('Pixel')
        plt.ylabel('Elevation')
    plt.tight_layout()
    plt.show()

def show_profiles_index(profile_list):
    """Display profiles with structural boundary points marked."""
    num_profiles, num_rows = len(profile_list), (len(profile_list) + 1) // 2
    plt.figure(figsize=(12, 8))
    for i, profile in enumerate(profile_list):
        i1, i2, i3, i4 = get_structure_boundaries(profile, len(profile))
        plt.subplot(num_rows, 2, i + 1)
        plt.plot(range(len(profile)), profile, color=f'C{i}')
        for idx in [i1, i2, i3, i4]:
            if idx < len(profile):
                plt.scatter(idx, profile[idx], color='red')
                plt.text(idx, profile[idx], f'({idx},{profile[idx]:.2f})', color='black', fontsize=11, ha='left')
    plt.tight_layout()
    plt.show()

def get_indexs(fh, length, avg_elevation):
    """Identify key structural indices in a profile."""
    fh = np.array(fh)
    l = len(fh)
    max_indices = [i for i in range(2, l - 1) if fh[i] >= fh[i - 1] and fh[i] > fh[i + 1]]
    max_indices_unique = np.unique(max_indices)
    index1, index4 = 0, length - 1

    if max_indices_unique.size > 0:
        best_idx_l = max([idx for idx in max_indices_unique if fh[idx] > fh[0] and idx < l // 4 and fh[idx] > avg_elevation], 
                        key=lambda x: fh[x], default=None)
        if best_idx_l:
            index1 = best_idx_l
        best_idx_r = max([idx for idx in max_indices_unique if fh[idx] > fh[-1] and idx > 3 * l // 4 and fh[idx] > avg_elevation], 
                        key=lambda x: fh[x], default=None)
        if best_idx_r:
            index4 = best_idx_r

    min_indices = [i for i in range(2, l - 1) if fh[i] < fh[i - 1] and fh[i] <= fh[i + 1]]
    min_indices_unique = np.unique(min_indices)
    index2, index3 = length // 2 - 1, length // 2 - 1
    best_score1, best_score2 = 0, 0

    for id in min_indices_unique:
        if index1 < id < l // 2:
            sl1 = fh[max(0, id - (l // 4))] - fh[id]
            if sl1 > best_score1:
                best_score1, index2 = sl1, id - 1
        if l // 2 < id < index4:
            sr1 = fh[min(id + (l // 4), l - 1)] - fh[id]
            if sr1 > best_score2:
                best_score2, index3 = sr1, id + 1
    return index1, index2, index3, index4

def get_structure_boundaries(horizontal_profile, length):
    """Get structural boundary indices for a profile."""
    avg_elevation = np.mean(horizontal_profile)
    return get_indexs(horizontal_profile, length, avg_elevation)

if __name__ == "__main__":
    """Main execution: load image and display profiles with boundaries."""
    img = '/home/xgq/Desktop/HF/yunshi/machine_learing/cropped_train_dem/special/06-0-001752.jpg'
    img_array = np.array(Image.open(img))
    p_list = get_profile(img_array, num_profile=4)
    show_profiles_index(p_list)
