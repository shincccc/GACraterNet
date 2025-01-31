from demo import get_segments
import numpy as np
import os
from sam.deal_segments import is_touching_edge
from PIL import Image

def remove_fp(pd_path, tp_path, fp_path, sam_seg_path):
    for pic in os.listdir(pd_path):
        img_path = os.path.join(pd_path, pic)
        c_array = np.array(Image.open(img_path))
        c_array_eh = enhance_dem_array(c_array, 3)
        floor_seg, largest_wall_region, crater_segment, centroid_x, \
        centroid_y = get_segments(c_array_eh, os.path.join(sam_seg_path, pic))
        if crater_segment is not None:
            processed_img = Image.fromarray(c_array_eh)
            processed_img.save(os.path.join(tp_path, pic))
        else:
            processed_img = Image.fromarray(c_array_eh)
            processed_img.save(os.path.join(fp_path, pic))

def enhance_dem_array(array, n):
    # 确保输入是二维数组
    if len(array.shape) != 2:
        array = array[:,:,0]
    # 计算最大值和最小值
    max_val = np.max(array)
    min_val = np.min(array)
    # 计算差δh
    delta_h = max_val - min_val
    # 计算a
    a = delta_h / n
    # 用每个像素值对a取余
    new_array = array % a
    # 归一化到0~255
    norm_array = (new_array - np.min(new_array)) / (np.max(new_array) - np.min(new_array)) * 255
    norm_array = norm_array.astype(np.uint8)  # 转换为uint8类型
    norm_array = np.stack([norm_array]*3, axis=-1)
    # print(norm_array.shape)
    return norm_array

if __name__ == "__main__":
    pd_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/exp_dem'
    tp_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/tp/'
    fp_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/fp/'
    sam_seg_path = '/home/xgq/Desktop/HF/yunshi/results/remove_fp/sam/'
    remove_fp(pd_path, tp_path, fp_path, sam_seg_path)