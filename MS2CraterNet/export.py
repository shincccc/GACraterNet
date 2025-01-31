from ultralytics import YOLO
model = YOLO('runs/detect/bi_3sff_dcn/train113/weights/best.pt')
results = model.export(format='onnx')