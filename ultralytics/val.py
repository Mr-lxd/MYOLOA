import sys
sys.path.insert(0, r"D:\DL_Project\YOLOv8-multi-task-main\ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO


def main():
    # model = YOLO('yolov8s-seg.pt')
    # number = 3 #input how many tasks in your work
    model = YOLO(r'D:\DL_Project\YOLOv8-multi-task-main\runs\20250406-n-DICS_Res_add_v11_1_Seg_1_DGCST_3\weights\best.pt')  # 加载自己训练的模型# Validate the model
    # metrics = model.val(data='/home/jiayuan/ultralytics-main/ultralytics/datasets/bdd-multi.yaml',device=[4],task='multi',name='v3-model-val',iou=0.6,conf=0.001, imgsz=(640,640),classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)  # no arguments needed, dataset and settings remembered

    metrics = model.val(data=r'D:\DL_Project\YOLOv8-multi-task-main\ultralytics\datasets\bdd-multi.yaml',device=[0],task='multi',name='20250407',
                        iou=0.7,conf=0.001, imgsz=(640,640),
                        classes=[0,1],
                        # combine_class=[2,3,4,9],
                        single_cls=False,
                        save_json=False,
                        project="D:/DL_Project/YOLOv8-multi-task-main/runs")  # no arguments needed, dataset and settings remembered
    # for i in range(number):
    #     print(f'This is for {i} work')
    #     print(metrics[i].box.map)    # map50-95
    #     print(metrics[i].box.map50)  # map50
    #     print(metrics[i].box.map75)  # map75
    #     print(metrics[i].box.maps)   # a list contains map50-95 of each category


if __name__ == "__main__":
    # 如果是打包环境（如PyInstaller），需要添加下面这行
    # multiprocessing.freeze_support()
    main()