import sys
sys.path.insert(0, "D:/DL_Project/YOLOv8-multi-task-main/ultralytics")

from ultralytics import YOLO


number = 2 #input how many tasks in your work
model = YOLO(r'D:\DL_Project\YOLOv8-multi-task-main\runs\20250406-n\weights\best.pt')  # Validate the model

# 假设你想保持相同的降采样比例
# scale_factor = 384/720  # 当前使用的imgsz与标准尺寸的比例
# new_h = int(3072 * scale_factor)
# new_w = int(4096 * scale_factor)

model.predict(source=r'D:\navigationData\ImagesForYOLOv8Mutil_3\images\val',
              device=[0],name='predict_20250408', save=False, project="D:/DL_Project/YOLOv8-multi-task-main/runs/predict",
              conf=0.001, iou=0.7,
              show_labels=False)
