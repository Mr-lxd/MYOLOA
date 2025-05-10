import sys
sys.path.insert(0, "D:/DL_Project/YOLOv8-multi-task-main/ultralytics")

from ultralytics import YOLO

def main():
    # Load model
    model = YOLO(
        'D:/DL_Project/YOLOv8-multi-task-main/ultralytics/models/v8/MYA.yaml',
        task='multi'
    )

    # Train the model
    model.train(
        data='D:/DL_Project/YOLOv8-multi-task-main/ultralytics/datasets/bdd-multi.yaml',
        batch=8,
        epochs=300,
        imgsz=(640,640),
        device=[0],
        name='20250406-n-v5',
        val=True,
        task='multi',
        classes=[0,1],
        # combine_class=[2,3,4,9],
        single_cls=False,
        project="D:/DL_Project/YOLOv8-multi-task-main/runs",
        augment=True,
        workers=8,
        amp=True,
        cache=True
    )

if __name__ == "__main__":
    # 如果是打包环境（如PyInstaller），需要添加下面这行
    # multiprocessing.freeze_support()
    main()