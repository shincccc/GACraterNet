from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.classify import ClassificationTrainer

args = dict(model='yolov8l.yaml', data='./ultralytics/cfg/datasets/coco.yaml', epochs=120, batch=10, workers=8)

trainer = DetectionTrainer(overrides=args)
trainer.train()


