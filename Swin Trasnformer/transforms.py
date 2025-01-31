# import os
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from tqdm import tqdm
# import time
# import torch
#
#
#
# def augment_and_save_images(dataset_dir, class_label_to_augment, num_augmentations=3, output_format='JPEG'):
#     A_transform = transforms.Compose([
#         transforms.RandomRotation(10),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
#         transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
#     ])

#     # 加载数据集
#     dataset = ImageFolder(root=dataset_dir, transform=None)

#     # 使用 tqdm 包装循环，显示进度条
#     for idx, (inputs, labels) in tqdm(enumerate(dataset), total=len(dataset)):
#         # 获取原图像文件路径
#         original_path = dataset.samples[idx][0]

#         # 判断图像的类别是否为指定类别
#         if labels == class_label_to_augment:
#             for i in range(num_augmentations):
#                 # 生成唯一的文件名
#                 timestamp = int(time.time() * 1000000)
#                 augmented_path = os.path.join(os.path.dirname(original_path), f'aug_{i + 1}_{timestamp}.jpg')

#                 # 进行数据增强并保存
#                 augmented_image = A_transform(inputs)
#                 augmented_image.save(augmented_path, format=output_format)


#     print(f"类别 {class_label_to_augment} 的数据增强并保存完成。")

# # 示例用法
# dataset_directory = '/root/autodl-tmp/deep-learning-for-image-processing-master/data_set/lunar_data/lunar_photos'
# class_label_to_augment = 4
# number_of_augmentations = 3
# output_image_format = 'JPEG'

# augment_and_save_images(dataset_directory, class_label_to_augment, number_of_augmentations, output_image_format)
#
from model import swin_tiny_patch4_window7_224 as create_model
import torch
import torch.nn as nn

# 创建模型实例
model = create_model()

# 将模型的分类头部替换为Identity层，以便模型输出特征而不是分类结果
model.avgpool = nn.Identity()
model.head = nn.Identity()

# 假设你有一个batch的图像数据
images = torch.randn(8, 3, 224, 224)  # 8张224x224的图像，3个颜色通道

# 确保模型处于评估模式，这通常在提取特征时需要
model.eval()

# 不计算梯度，因为在特征提取时通常不需要反向传播
with torch.no_grad():
    # 通过模型获取特征
    features = model(images)

print(features.shape)  # 输出特征的形状
