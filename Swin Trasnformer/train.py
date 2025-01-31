import os
import argparse
import json
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate
from torchvision.transforms import functional as F


class RandomResizedCropAndRotate(transforms.RandomResizedCrop):
    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

        # 随机旋转
        angle = transforms.RandomRotation.get_params([-45, 45])
        img = F.rotate(img, angle)

        # 随机裁剪
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.8, 1.0), ratio=(0.8, 1.2))
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

        return img
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # tb_writer = SummaryWriter()

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 256
    # 修改数据增强的变换
    data_transform = {
        "train": transforms.Compose([
            RandomResizedCropAndRotate(img_size),  # 自定义增强
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.3174, 0.3171, 0.3175], [0.1362, 0.1362, 0.1363])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.3174, 0.3171, 0.3175], [0.1362, 0.1362, 0.1363])
        ])
    }
    # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))+'/yunshi/swin_transformer/'  # get data root path
    print("data_root:", data_root)
    #image_path = os.path.join(data_root, "cropped_train_image", "data_set")  # flower data set path
    image_path = os.path.join(data_root, "dataset_eh_1009")  # flower data set path
    #image_path = os.path.join(data_root, "cls_dataset")  # flower data set path


    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    lunar_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in lunar_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
    # collate_fn = train_dataset.collate_fn

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)

    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw )
    # collate_fn = validate_dataset.collate_fn
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0  # Track the best validation accuracy
    for epoch in range(args.epochs):
        scheduler.step()
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # Save the model only if the current validation accuracy is better than the best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./weights/morf_best_model-2.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/home/zx3srs/yimin/zyn/deep-learning-for-image-processing-master/data_set/lunar_data/lunar_photos")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/home/xgq/Desktop/HF/yunshi/swin_transformer/swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    # parser.add_argument('--weights', type=str, default='',
    #                     help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
