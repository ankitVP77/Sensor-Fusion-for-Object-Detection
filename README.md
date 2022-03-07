# Sensor-Fusion-for-Object-Detection
This project focuses on performing early sensor fusion of raw camera and lidar data from the KITTI dataset to faciliate detecting objects and estimating their depth information.

<img src="deliv/out_1.gif" width=1080px>
<img src="deliv/out_2.gif" width=1080px>
<img src="deliv/out_3.gif" width=1080px>
<img src="deliv/out_4.gif" width=1080px>

## Packages required
1) Numpy
2) OpenCV 4
3) Matplotlib
4) Yolov4 (pip install yolov4==2.0.2)
5) Tensorflow 2
6) Open3d


## How to run code

1) The data folder contains 5 images and corresponding lidar data from the KITTI Vision dataset (You can use your own dataset as well)
2) The yolov4 folder contains the tiny-yolov4 weights. The original yolov4 weights can be downloaded and added to the same folder for use
3) In the ```early_fusion.py``` file change the index variable to index to the different images in the data folder.
4) The ```YoloOD``` class takes a tiny_model initialization parameter. Change this to true if you want to use tiny yolo else let it be false
5) For results run the ```early_fusion.py``` file.
