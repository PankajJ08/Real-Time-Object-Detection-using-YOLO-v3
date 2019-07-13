# Real-Time-Object-Detection-using-YOLO-v3

>This project implements a real time object detection via webcam and image detection using YOLO algorithm. YOLO is a object detection algorithm which stand for You Only Look Once. I've implemented the algorithm from scratch in PyTorch using pre-trained weights.
YOLOv3 was published in research paper: [YOLOv3: An Incremental Improvement: Joseph Redmon, Ali Farhadi](https://pjreddie.com/media/files/papers/YOLOv3.pdf). It's originally implemented in [YOLOv3](https://github.com/pjreddie/darknet).

>COCO dataset is used for training.
 
>Real time detection can be use via command prompt or GUI.
 
---

## How to use?
1. Clone the repository

  `git clone https://github.com/PankajJ08/Real-Time-Object-Detection-using-YOLO-v3.git`
  
2. Move to the directory

  `cd Real-Time-Object-Detection-using-YOLO-v3`
  
3. To infer on an image that is stored on your local machine

  `python3 img_detect.py --image path='/path/to/image/'`

4. To use in real-time on webcam

  `python3 camera.py`
  
5. To use GUI for real-time detection

  `python gui.py`
  
  ** Download the official [weight file](https://pjreddie.com/media/files/yolov3.weights) and place it under a folder called weight. 
 
---

 ## Graphical User Interface
  
  ![GUI…](https://user-images.githubusercontent.com/26256765/61166968-e2a7e280-a554-11e9-8385-fe75e0f0ac54.png)

 ---  
 
 ## References
 1. [Paperspace: YOLO object detector in PyTorch](https://blog.paperspace.com/tag/series-yolo/)
