import os


from ultralytics import YOLO


ROOT = os.path.dirname(os.path.dirname(__file__))

DIR_DATA = os.path.join(ROOT, "data", "yolo_ant", "data.yaml")
DIR_MODEL = os.path.join(ROOT, "models", "yolov8m.pt")


model = YOLO(DIR_MODEL)

model.__dict__


model.train(data=DIR_DATA, batch=16, imgsz=640, device="mps", save_period=1)
