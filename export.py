from ultralytics import YOLO

def export():
    model = YOLO('runs/multi_V1/weights/best.pt')  # 加载模型
    model.export(format='onnx',  #导出onnx格式，默认原路径保存，例如:best.onnx
                 imgsz=640,  # 模型的图片输入尺寸
                 dynamic=False,  # 禁止模型动态的图片输入大小
                 )

if __name__ == '__main__':
    export()
