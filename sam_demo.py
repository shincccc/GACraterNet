from PIL import Image
import numpy as np
from sam.segment_crater import segment_crater, extract_segments, evaluate_block
from sam.deal_segments import remove_noises, is_touching_edge
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import rasterio
import random
from osgeo import gdal
import cv2
from dem_enhance import enhance_dem_array


def find_bounding_box(binary_array):
    rows = np.any(binary_array, axis=1)
    cols = np.any(binary_array, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return min_col, min_row, max_col + 1, max_row + 1

def show_seg(segment):
    seg = np.array(segment * 255, dtype=np.uint8)
    seg = np.stack([seg] * 3, axis=-1)
    seg_img = Image.fromarray(seg)
    seg_img.show()

def generate_colors(anns, num_colors):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    colors = np.random.rand(num_colors, 3)
    colors[:, 0] = (colors[:, 0] - colors[:, 0].min()) / (colors[:, 0].max() - colors[:, 0].min())
    colors[:, 1] = (colors[:, 1] - colors[:, 1].min()) / (colors[:, 1].max() - colors[:, 1].min())
    colors[:, 2] = (colors[:, 2] - colors[:, 2].min()) / (colors[:, 2].max() - colors[:, 2].min())
    return colors, sorted_anns

def show_anns(sorted_anns, colors, sam_seg_path):
    # 获取最大的分割区域形状
    max_shape = sorted_anns[0]['segmentation'].shape
    # 创建一个全透明的背景图0像
    img = np.zeros((max_shape[0], max_shape[1], 4), dtype=np.float32)
    img[:, :, 3] = 1  # 设置 alpha 通道为 1（不透明）
    # 生成颜色列表
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([colors[i], [0.35]], axis=0)
        img[m == 1] = color_mask  # 假设分割区域的值为 1

    # 将图像转换为适合保存的格式
    img_bgr = img[:, :, :3]  # 提取 RGB 通道
    img_bgr = (img_bgr * 255).astype(np.uint8)  # 转换为 uint8
    # 显示图像
    cv2.imwrite(sam_seg_path, img_bgr)  # 保存为图
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

def compute_centroid(segment):
    # Get the coordinates of the pixels that are part of the foreground (value = 1)
    y_indices, x_indices = np.where(segment == 1)  # y: row indices, x: column indices

    if len(y_indices) == 0 or len(x_indices) == 0:
        # If the segment is empty (no foreground pixels), return None
        return None

    # Calculate the centroid coordinates
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)

    return centroid_x, centroid_y

def get_segments(img_array, sam_seg_path):
    h, w = img_array.shape[:2]
    # diam = (h + w) // 2
    s = h * w

    # Segment the crater
    masks = segment_crater(img_array)

    colors, sorted_anns = generate_colors(masks, len(masks))
    show_anns(sorted_anns, colors, sam_seg_path)

    # Extract segments
    internal_blocks_info, crater_wall_info, wall_segments = extract_segments(w, h, masks, min_area_threshold=s / 10)

    if len(internal_blocks_info) == 0:
        return None, None, None, w/2, h/2
    else:
        floor_centroid, floor_area, floor_seg = internal_blocks_info[0]
        centroid_y, centroid_x = floor_centroid

        if wall_segments:
            # print("have walls")
            # Combine wall segments
            combined_walls = np.sum(wall_segments, axis=0)
            combined_walls = np.array(combined_walls * 255, dtype=np.uint8)
            labels = measure.label(combined_walls, connectivity=2)
            # Extract properties
            props = measure.regionprops(labels)
            if props:
                max_area = max(prop.area for prop in props)
                max_label = next(prop.label for prop in props if prop.area == max_area)
                largest_wall_region = np.where(labels == max_label, 1, 0)
                crater_segment = np.logical_or(floor_seg, combined_walls)
            else:
                largest_wall_region = None
                crater_segment = floor_seg
        else:
            # print("no walls")
            largest_wall_region = None
            crater_segment = floor_seg

        crater_segment = remove_noises(crater_segment)
        # crater_segment_img = Image.fromarray(crater_segment)
        # crater_segment_img.show()

        return floor_seg, largest_wall_region, crater_segment, centroid_x, centroid_y

def get_crs(tiff_path):
    crs = get_crs_from_tiff(tiff_path)
    return crs

def save_to_shapefile(polygons, shp_save_path, crs):
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf.to_file(shp_save_path, driver='ESRI Shapefile')

def get_crs_from_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:
        return src.crs.to_string()

def preprocess_array(arr):
    struct = generate_binary_structure(2, 2)
    arr = binary_opening(arr, structure=struct).astype(int)  # Change here from np.int to int
    arr = binary_closing(arr, structure=struct).astype(int)  # Change here from np.int to int
    return arr

def array_to_polygon(arr, tile_lu_x, tile_lu_y, patch_lu_x, patch_lu_y):
    if arr is None:
        return None

    arr = preprocess_array(arr)  # 预处理数组以去噪和平滑

    rows, cols = arr.shape
    exterior_coords = []
    grid_size = 50

    lu_x = tile_lu_x + patch_lu_x*50
    lu_y = tile_lu_y - patch_lu_y*50

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
                # 当前单元格及其邻居
            current = arr[i][j]
            neighbors = [arr[i - 1][j], arr[i + 1][j], arr[i][j - 1], arr[i][j + 1]]
            # 检查是否在边界上
            if (current == 1 and sum(neighbors) == 2) or (current == 0 and sum(neighbors) == 3):
                    # 转换为Shapely的点对象
                x, y = lu_x + j * grid_size, lu_y - i * grid_size
                exterior_coords.append((x, y))

    # 通过set去重，并转化为点对象
    if exterior_coords:
        unique_coords = list(set(exterior_coords))
        polygon = Polygon(unique_coords).convex_hull  # 使用凸包简化形状
        simplified_polygon = polygon.simplify(1.0, preserve_topology=False)  # 平滑多边形
        return simplified_polygon
    else:
        return None

def get_geo_cord(col_nums, patch_id, x, y):  # patch_id=144
    patch_size = 1024
    x_geo = (patch_id % col_nums) * patch_size + x
    y_geo = (patch_id // col_nums) * patch_size + y
    return x_geo, y_geo

def expand_window(tile_array, patch_id, x_min, y_min, w0, h0, w, h, centroid_x, centroid_y):
    x_bia = centroid_x - w // 2
    y_bia = centroid_y - h // 2
    print("x_bia, y_bia:", x_bia, y_bia)

    col_nums = 13
    tile_w = 13336
    tile_h = 17782
    diam = (w0 + h0)//2
    x_max = int(x_min + w0)
    y_max = int(y_min + h0)
    x_tile_min, y_tile_min = get_geo_cord(col_nums, patch_id, x_min, y_min)
    x_tile_max, y_tile_max = get_geo_cord(col_nums, patch_id, x_max, y_max)
    x_tile_min, y_tile_min, x_tile_max, y_tile_max = int(x_tile_min - int(diam * 0.2)), int(
        y_tile_min - int(diam * 0.2)), int(
        x_tile_max + int(diam * 0.2)), int(y_tile_max + int(diam * 0.2))
    x_geo_min = max(x_tile_min, 0)
    y_geo_min = max(y_tile_min, 0)
    x_geo_max = min(x_tile_max, tile_w)
    y_geo_max = min(y_tile_max, tile_h)

    if x_bia >= 0 and y_bia >= 0:
        x_min_n = x_geo_min
        y_min_n = y_geo_min
        x_max_n = min(round(x_geo_max + x_bia), tile_w)
        y_max_n = min(round(y_geo_max + y_bia), tile_h)
    elif x_bia >= 0 and y_bia < 0:
        x_min_n = x_geo_min
        y_min_n = max(round(y_geo_min + y_bia), 0)
        x_max_n = min(round(x_geo_max + x_bia), tile_h)
        y_max_n = y_geo_max
    elif x_bia < 0 and y_bia >= 0:
        x_min_n = max(round(x_geo_min + x_bia), 0)
        y_min_n = y_geo_min
        x_max_n = x_geo_max
        y_max_n = min(round(y_geo_max + y_bia), tile_h)
    elif x_bia < 0 and y_bia < 0:
        x_min_n = max(round(x_geo_min + x_bia), 0)
        y_min_n = max(round(y_geo_min + y_bia), 0)
        x_max_n = x_geo_max
        y_max_n = y_geo_max

    if y_min_n == 0:
        y_max_n = y_min_n + h
    elif y_max_n == tile_h:
        y_min_n = y_max_n - h
    if x_min_n == 0:
        x_max_n = x_min_n + w
    elif x_max_n == tile_w:
        x_min_n = x_max_n - w

    print("w, h, y_min_n, y_max_n, x_min_n, x_max_n:", w, h, y_min_n, y_max_n, x_min_n, x_max_n)

    new_cropped_array = tile_array[y_min_n:y_max_n, x_min_n:x_max_n]
    new_cropped_array = np.stack([new_cropped_array]* 3, axis=-1)
    floor_seg, largest_wall_region, crater_segment, centroid_x, centroid_y = get_segments(new_cropped_array)

    return x_min_n, y_min_n, x_max_n, y_max_n, floor_seg, largest_wall_region, crater_segment, centroid_x, centroid_y

if __name__ == "__main__":

    #205_HMC_13E30_co5_210_4 简单坑 偏移
    #186_HMC_13E10_co5_121_3 复杂坑 平底 偏移
    # 简单 过小
    #sam_seg_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/sam/169_HMC_13E20_co5_111_0.jpg'
    #img_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/fp/169_HMC_13E20_co5_111_0.jpg'

    sam_seg_path = '/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/sam/73_HMC_13E10_co5_96_0.jpg'
    img_path = '/home/xgq/Desktop/HF/yunshi/results/exp_dem_sam_2/wsb/73_HMC_13E10_co5_96_0.jpg'

    # img_path = '/home/xgq/Desktop/HF/yunshi/results/exp_dem_uint8/29_HMC_13E30_co5_31_0.jpg'

    image = Image.open(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_array = np.array(image)
    img_array = enhance_dem_array(img_array, n=5)
    img_array = np.stack([img_array]*3, axis=-1)

    h, w = img_array.shape[:2]
    floor_seg, largest_wall_region, crater_segment, centroid_x, centroid_y = get_segments(img_array, sam_seg_path)




