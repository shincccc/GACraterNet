import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
#from model import swin_tiny_patch4_window7_224 as create_model
from swin_transformer.model import swin_tiny_patch4_window7_224 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 256
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "/home/xgq/Desktop/HF/yunshi/swin_transformer/cropped_train_image/data_set/train/CPKs/06-0-000322.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=4).to(device)
    # load model weights
    model_weight_path = "/home/zx3srs/yimin/zyn/deep-learning-for-image-processing-master/pytorch_classification/swin_transformer/weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


def predict_crater(img_array, model_weight_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    json_path = '/home/xgq/Desktop/HF/yunshi/swin_transformer/class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # print("img_array:", img_array)
    img_size = 256  # 256
    # img_array = np.transpose(img_array, (1, 2, 0))
    # print(img_array.shape)
    if img_array.dtype == np.float32:
        img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.143)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.3174, 0.3171, 0.3175], [0.1362, 0.1362, 0.1363])])

    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model = create_model(num_classes=4).to(device)
    model_weight_path = '/home/xgq/Desktop/HF/yunshi/swin_transformer/weights/morf_best_model-2.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    return class_indict[str(predict_cla)]


if __name__ == '__main__':
    main()
