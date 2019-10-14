#intall ImageAI and its dependies

pip install numpy
pip install scipy
pip install keras
pip install opencv
pip install pillow
pip install matplotlib
pip install h5py
pip install tensorflow

pip3 install imageai --
upgrade

#Download the RetinaNet model file- the object detection model that will be used here.

from imageai.Detection import ObjectDetection
import os


execution_path= os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image= os.path.join(execution_path, "image.jpg"), output_image_path=os.join(execution_path, "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"], ":", eachObject["percentage_probability"])
