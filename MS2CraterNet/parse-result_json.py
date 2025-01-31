import json

# JSON文件的路径
file_path = '/home/xgq/Desktop/HF/yunshi/ultralytics-m/ultralytics-modify/runs/detect/val8/predictions.json'

# 从文件中读取JSON数据
with open(file_path, 'r') as file:
    data = json.load(file)

# 创建一个新的字典来存储每个图片的边界框
image_dict = {}

# 遍历每一个元素
for item in data:
    image_id = item['image_id']
    bbox = item['bbox']

    # 如果图片已存在于字典中，添加新的bbox到该图片的列表中
    if image_id in image_dict:
        image_dict[image_id].append(bbox)
    else:
        image_dict[image_id] = [bbox]

# 计算总的bbox数量
total_bboxes = sum(len(bboxes) for bboxes in image_dict.values())

# 输出字典和总bbox数量
# print("Dictionary with image IDs and bounding boxes:")
# print(image_dict)
print("Total number of bounding boxes:", total_bboxes)

# 指定要保存的文件路径
save_path = '/home/xgq/Desktop/HF/yunshi/results/exp_image_data.json'

# 将字典保存为JSON文件
with open(save_path, 'w') as file:
    json.dump(image_dict, file, indent=4)

print("Dictionary saved to:", save_path)