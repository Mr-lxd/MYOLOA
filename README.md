
# YOLO-SSR: A Lightweight Model for Synchronized Crop Stem Detection and Row Segmentation  
  
**YOLO-SSR (YOLO for Synchronized Stem-Row)** is a lightweight deep learning model designed for simultaneous crop stem detection and crop row segmentation, specifically tailored for agricultural navigation line extraction. This work explores the synergistic contribution of these two tasks to enhance autonomous navigation.
  
This project is based on and extends the work of [You Only Look at Once for Real-time and Generic Multi-Task](https://github.com/JiayuanWang-JW/YOLOv8-multi-task). We express our gratitude to the original authors for their foundational work.

![YOLO-SSR.jpg](pic/YOLO-SSR.jpg)

## Model Architecture Highlights  
- **Tri-Path Adaptive Convolution (TriPAC):** A novel module integrated into the backbone for efficient multi-scale feature capture.  
- **Enhanced Detection Branch:** Incorporates Space-to-Depth Convolution (SPDC) for shallow feature enhancement and Dynamic Group Shuffle Transformer (DGST) for optimized contextual information.  
- **Optimized Segmentation Branch:** Features a minimalist neck and head, reusing features from the detection branch and incorporating TriPAC.
  
| Model                   | AP<sub>50<sub> | AP<sub>75<sub> | AP<sub>50-95<sub> | AP<sub>s<sub> | AP<sub>m<sub>  | AP<sub>l<sub>  | AP<sub>mc<sub> | AP<sub>vt<sub> | FPS   | Inference Time(ms) | parameters | GFLOPs |
| :---------------------- | :--: | :--: | :-----: | :-: | :--: | :--: | :--: | :--: | :---: | :----------------: | :--------: | :----: |
| YOLOv8n                 | 59.2 | 13.5 | 23.2    | 1   | 17   | 35.9 | 23.2 | 33.8 | 131.6 | 7.6                | 3.01M      | 8.1    |
| YOLOv12n                | 57.9 | 13.6 | 23.1    | 0.7 | 16.4 | 36.9 | 23.1 | 34.2 | 526.3 | 1.9                | 2.56M      | 6.3    |
| DINO(ResNet50)          | 64.7 | 12.7 | 24.5    | 2.3 | 19.5 | 34.8 | 24.5 | 32   | 19.5  | 51.3               | 47.54M     | 235    |
| Dynamic R-CNN(ResNet50) | 30.7 | 4.3  | 10.8    | 0   | 8.5  | 16.7 | 11.6 | 0.8  | 22.1  | 45.2               | 41.348M    | 178    |
| RTMDet(CSPNeXt)         | 62   | 14.7 | 25.1    | 0.7 | 18.4 | 38.8 | 25   | 33.8 | 77.6  | 12.9               | 4.873M     | 8.025  |
| TOOD(ResNet50)          | 46.9 | 6.3  | 16      | 0.3 | 11.3 | 25.7 | 16   | 17.9 | 17.4  | 57.5               | 32.018M    | 168    |
| VarifocalNet(ResNet50)  | 46.2 | 5.4  | 15.3    | 0.7 | 10.2 | 25.3 | 15.3 | 22   | 22.8  | 43.9               | 32.709M    | 161    |
| ours(det)               | 61.7 | 15.3 | 24.9    | 1.2 | 18.9 | 38.1 | 24.9 | 29.4 | 202   | 4.95               | 2.45M      | 10.5   |

  | Model                     | MaskAP<sub>50<sub> | MaskAP<sub>75<sub> | MaskAP<sub>50-95<sub> | mIoU  | L<sub>d_mean</sub> | L<sub>d_median</sub> | L<sub>d_std</sub> | FPS   | Inference Time(ms) | parameters | GFLOPs |
| :------------------------ | :------: | :------: | :---------: | :---: | :-------------: | :---------------: | :-----------: | :---: | :----------------: | :--------: | :----: |
| YOLOv8n-seg               | 99.6     | 81.5     | 64.8        | 84.67 | 33.35           | 24.85             | 19.08         | 100   | 10                 | 3.26M      | 12     |
| YOLOv12n-seg              | 98.9     | 77.6     | 63.3        | 84.94 | 34.79           | 26.68             | 21.02         | 344.8 | 2.9                | 2.81M      | 10.2   |
| SOLOv2(ResNet18)          | 98.9     | 77.9     | 62.7        | 88.68 | 34.9            | 26.12             | 19.36         | 29.7  | 33.7               | 18.09M     | 42.491 |
| SOLOv2(ResNet101)         | 97.7     | 64.2     | 58          | 87.38 | 39.14           | 26.37             | 22.31         | 23.7  | 42.2               | 65.221M    | 282    |
| RTMDet-ins(CSPNeXt)       | 99.4     | 81.1     | 64.3        | 70.37 | 33.38           | 24.75             | 20.07         | 100.6 | 9.9                | 5.615M     | 11.873 |
| Mask R-CNN(ConvNeXt V2)   | 99.6     | 75.2     | 64.9        | 85.42 | 33.75           | 24.57             | 20.8          | 16.1  | 62.1               | 108M       | 421    |
| SparseInst(ResNet50)      | 90.6     | 10       | 34.5        | 37.75 | 52.96           | 34.46             | 24.09         | 24.2  | 41.3               | 31.617M    | 99.22  |
| ours(ins)                 | 93.7     | 76.8     | 60.3        | 88.72 | 34.18           | 24.14             | 19.23         | 202   | 4.95               | 2.45M      | 10.5   |  
  
## Visual Results

![visualization.jpg](pic/visualization.jpg)
  
## Dataset  
This study introduces the **SeedlingStemRow (SSR) dataset**, comprising field images of sugarcane, corn, and rice during seedling stages, with annotations for crop stems and crop rows.  
- **Total Images:** 758 high-resolution images  
- **Annotations:** 25,419 crop stem annotations and 758 crop row segmentation annotations.  
- **Availability:** The dataset is publicly available on Kaggle:  
[SSR Dataset on Kaggle](https://www.kaggle.com/datasets/xxdxdxd/seedlingstemrow)  