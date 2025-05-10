import sys
sys.path.insert(0, "D:/DL_Project/YOLOv8-multi-task-main/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
# model = YOLO('ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml', task='multi')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('D:/DL_Project/YOLOv8-multi-task-main/ultralytics/models/v8/yolov8-bdd-v3-one.yaml', task='multi').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='D:/DL_Project/YOLOv8-multi-task-main/ultralytics/datasets/bdd-multi.yaml', workers=4, batch=2, epochs=300, imgsz=(640,640), device=[0], name='multi_V1', val=True, task='multi',classes=[0,1,2],combine_class=[2,3,4,9],single_cls=False)

