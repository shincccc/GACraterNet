import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, LineString, MultiPolygon
from skimage import measure
from shapely.ops import unary_union
from scipy.ndimage import label as label2
import cv2
from PIL import Image

def show_seg0(segment):
    seg = np.array(segment * 255, dtype=np.uint8)
    seg = np.stack([seg] * 3, axis=-1)
    seg_img = Image.fromarray(seg)
    seg_img.show()

def show_segment_list(segments):
    for segment in segments:
        show_seg0(segment)

def resolve_overlaps(segments):
    num_segments = len(segments)
    areas = [np.sum(segment) for segment in segments]  # 计算每个分割块的面积
    ordered_indices = np.argsort(areas)[::-1]
    ordered_indices = ordered_indices.tolist()  # 转换为列表格式
    # 初始化最终的分割结果，全部为False
    final_segments = []
    # 遍历排序后的分割块索引
    for index in ordered_indices:
        segment = segments[index].copy()  # 使用copy()来避免修改原始的segments数据
        for other_index in ordered_indices:
            if index == other_index:
                # print("index == other_index")
                continue
            other_segment = segments[other_index]
            overlap = np.logical_and(segment, other_segment)  # 找到重叠区域
            # 此时， 也要将other_segment加入final_segments，并将other_segment从ordered_indices中移除
            if np.sum(overlap) == 0: #无重叠
                # print("no overlap")
                continue
            elif np.sum(segment) >= np.sum(other_segment): #包含了其他块
                # print("与其他块重叠")
                # print(np.where(overlap==1))
                # 将重叠部分去掉
                # show_segment_list([segment, other_segment])
                segment[overlap] = False
            elif np.sum(overlap) < np.sum(other_segment): #被其他块包含
                # print("与其他块重叠")
                continue
            else:  #不完全重叠或的情况，将重叠部分从面积更大的一方去掉
                continue
        final_segments.append(segment)

    return final_segments

def calculate_centroids(masks):
    centroids = []

    for mask in masks:
        # 找出所有True值的索引
        true_indices = np.argwhere(mask)

        # 检查是否有True值（即分割块是否为空）
        if true_indices.size > 0:
            # 计算x和y坐标的和
            x_sum = true_indices[:, 1].sum()
            y_sum = true_indices[:, 0].sum()

            # 计算True值的数量（即面积）
            num_true = len(true_indices)

            # 计算质心（x和y坐标的平均值）
            centroid_x = x_sum / num_true
            centroid_y = y_sum / num_true

            # 将质心添加到列表中
            centroids.append((centroid_x, centroid_y))
        else:
            # 如果分割块为空，可以添加一个None或者特殊值来表示
            centroids.append(None)

    return centroids

def show_different(filtered_segments, filtered_segments_or):
    # 检查两个列表的长度是否相同
    if len(filtered_segments) != len(filtered_segments_or):
        raise ValueError("filtered_segments 和 filtered_segments_or 必须具有相同的长度。")

    num_segments = len(filtered_segments)
    plt.figure(figsize=(10 * num_segments, 10))  # 根据分割块数量调整图像大小

    # 遍历分割块列表并可视化每一对处理前后的分割块
    for i in range(num_segments):
        plt.subplot(1, num_segments * 2, i * 2 + 1)  # 2n+1 为原图的位置
        plt.imshow(filtered_segments[i], cmap='gray')
        plt.title('Processed Segment {}'.format(i + 1))
        plt.axis('off')

        plt.subplot(1, num_segments * 2, i * 2 + 2)  # 2n+2 为处理后的图的位置
        plt.imshow(filtered_segments_or[i], cmap='gray')
        plt.title('Original Segment {}'.format(i + 1))
        plt.axis('off')

    plt.tight_layout()  # 调整子图布局以适应空间
    plt.show()

def show_bg_segments(segments):
    num_segments = len(segments)

    # 计算需要的行数以避免子图重叠
    num_rows = int(np.ceil(np.sqrt(num_segments)))

    plt.figure(figsize=(10 * num_rows, 10 * num_rows))  # 根据分割块数量调整图像大小

    # 遍历分割块列表并可视化
    for i, segment in enumerate(segments):
        plt.subplot(num_rows, num_rows, i + 1)
        plt.imshow(segment, cmap='gray', vmin=0, vmax=1)  # 使用灰度色图，vmin和vmax设置为0和1以确保白色为分割块，黑色为背景
        plt.title('Segment {}'.format(i + 1))
        plt.axis('off')  # 不显示坐标轴

    plt.tight_layout()  # 调整子图布局以适应空间
    plt.show()

def is_touching_edge(segment):
    if segment is None:
        return False
    # 检查上边缘
    if np.any(segment[0, :]):
        return True
    # 检查下边缘
    if np.any(segment[-1, :]):
        return True
    # 检查左边缘
    if np.any(segment[:, 0]):
        return True
    # 检查右边缘
    if np.any(segment[:, -1]):
        return True
    return False

def touching_edge_nums(segment):
    # 初始化触碰边缘的计数
    nums = 0

    # 检查四个边缘
    edges = {
        "top": segment[0:5, :],
        "bottom": segment[-5:, :],
        "left": segment[:, 0:5],
        "right": segment[:, -5:]
    }

    for edge in edges.values():
        if np.any(edge):
            nums += 1

    # print("edge nums:", nums)
    return nums


def remove_noises(segment):
    # 标记连通区域
    labeled_array, num_features = label2(segment, structure=np.ones((3, 3)))

    if num_features == 0:
        return segment  # 如果没有发现连通区域，返回原数组

    # 计算每个连通区域的大小
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # 忽略背景

    # 找到最大连通区域的标签
    max_label = sizes.argmax()

    # 保留最大连通区域，其他区域置为0
    cleaned_segment = np.where(labeled_array == max_label, 1, 0)

    return cleaned_segment


def bounding_box_aspect_ratio(segment):
    # 标签化分割块
    labeled_segment = label(segment)
    props = regionprops(labeled_segment)

    # 计算包围框的长宽比
    if props:
        bbox = props[0].bbox  # 获取第一个区域的包围框
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        return width / height if height > 0 else 0
    return None

def closest_aspect_ratio(target_ratio, ratios):
    # 找到最接近目标长宽比的索引
    return min(range(len(ratios)), key=lambda i: abs(ratios[i] - target_ratio))


# def is_shape_convex(segment):
#     # 确保 segment 是 uint8 类型
#     show_seg(segment)
#     if segment.dtype != np.uint8:
#         segment = (segment * 255).astype(np.uint8)
#
#     # 提取轮廓
#     contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if len(contours) == 0:
#         return None  # 没有找到轮廓
#
#     for contour in contours:
#         # 计算轮廓的面积
#         area = cv2.contourArea(contour)
#
#         # 计算凸包
#         hull = cv2.convexHull(contour)
#         hull_area = cv2.contourArea(hull)
#
#         # 计算面积比
#         area_ratio = area / hull_area if hull_area > 0 else 0
#
#         # 判断形状的凹凸性
#         if area_ratio < 1:
#             print("concave")
#             return "concave"
#         else:
#             print("convex")
#             return "convex"
#
#     return None  # 处理多轮廓的情况

def show_seg(segment):
    # Helper function to display images
    plt.imshow(segment, cmap='gray')
    plt.axis('off')
    plt.show()


# def is_shape_convex(segment):
#     # 确保 segment 是 uint8 类型
#
#     segment = remove_noises(segment)
#
#     if segment.dtype != np.uint8:
#         segment = (segment * 255).astype(np.uint8)
#
#     # 提取轮廓
#     contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if len(contours) == 0:
#         return None  # 没有找到轮廓
#
#     for contour in contours:
#         # 计算轮廓的面积
#         area = cv2.contourArea(contour)
#
#         # 计算凸包
#         hull = cv2.convexHull(contour)
#         hull_area = cv2.contourArea(hull)
#
#         # 计算面积比
#         area_ratio = area / hull_area if hull_area > 0 else 0
#
#         # 判断形状的凹凸性
#         if area_ratio < 0.8:
#             #print("concave")
#             convexity = "concave"
#         else:
#             #print("convex")
#             convexity = "convex"
#         # 可视化轮廓和凸包
#         # 创建一个三通道图像用于显示凸包
#         # hull_visualization = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
#         # 绘制原始轮廓为绿色，凸包为红色
#         # cv2.drawContours(hull_visualization, [contour], -1, (0, 255, 0), 2)
#         # cv2.drawContours(hull_visualization, [hull], -1, (0, 0, 255), 2)
#         # 显示图像
#         # show_seg(hull_visualization)
#
#         return convexity
#
#     return None  # 处理多轮廓的情况

def is_shape_convex(segment):
    # 确保 segment 是 uint8 类型
    segment = remove_noises(segment)

    if segment.dtype != np.uint8:
        segment = (segment * 255).astype(np.uint8)

    # 提取轮廓
    contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None  # 没有找到轮廓

    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 计算凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        # 计算面积比
        area_ratio = area / hull_area if hull_area > 0 else 0

        # 判断形状的凹凸性
        if area_ratio < 0.8:
            convexity = "concave"
        else:
            convexity = "convex"

        # 计算包围盒
        x, y, w, h = cv2.boundingRect(hull)
        bounding_box = ((x, y), (x + w, y + h))  # 左上角和右下角坐标

        # 输出凸包的轮廓和包围盒
        # print("Convexity:", convexity)
        # print("Convex Hull:", hull)
        # print("Bounding Box:", bounding_box)

        return convexity, hull, bounding_box

    return None, None, None  # 处理多轮廓的情况