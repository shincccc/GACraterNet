from ultralytics import YOLO

if __name__ == "__main__":
    pth_path = 'runs/detect/bi_3sff_dcn/train/weights/best.pt'
    #pth_path = "runs/detect/bi_3sff_nodcn/train/weights/best.pt"
    #pth_path = "runs/detect/bi/train/weights/best.pt"

    test_path = r"/home/xgq/Desktop/HF/yunshi/data/coco_rgb/val2017"
    #test_path = "/home/xgq/Desktop/HF/yunshi/data/experiment/val2017"
    model = YOLO(pth_path)  # load a custom model

    metrics = model.val(imgsz=640, conf=0.25, iou=0.5, save_json=True)
    print(f"mAP50-95: {metrics.box.map}")  # map50-95
    print(f"mAP50: {metrics.box.map50}")  # map50
    print(f"mAP75: {metrics.box.map75}")  # map75
    speed_metrics = metrics.speed
    total_time = sum(speed_metrics.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}")  # FPS

    #Predict with the model
    results = model(source = test_path, save=True, conf=0.25)  # predict on an image
    num = 0
    for result in results:
        boxes = result.boxes
        num += len(boxes)
    print(num)

