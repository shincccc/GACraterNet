from PIL import Image
import os

def resize_images_in_folder(folder_path, size=(256, 256)):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img = img.resize(size, Image.LANCZOS)
                        img.save(file_path)
                        print(f"Resized and saved: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")


# 使用示例
folder_path = '/home/xgq/Desktop/HF/yunshi/swin_transformer/dataset_eh/val'
resize_images_in_folder(folder_path)
