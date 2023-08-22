# TensorFlow_Object-Detection_API_Implementation
This repository uses the TensorFlow Object Detection API's pre-trained models to draw bounding boxes over 90 different types of objects. 

## Downloading a Pre-Trained Model
To download a pre-trained object detection model, visit TensorFlow's model zoo. There are many model trained on the COCO 2017 dataset. 
The code above uses the SSD MobileNet V2 FPNLite 320x320 model, but feel free to use any other model (just make sure to change the name and path accordingly in the code)

## Usage
In order to use the code above, you need to download the Object Detection API. 
You can either use these terminal commands: 

cd models/research/ 
protoc object_detection/protos/*.proto --python_out=. 
cp object_detection/packages/tf2/setup.py . 
python -m pip install .

You might see some errors and the modules might not download. If that does happen, just move or make a copy of the **object_detection** and **official** folders from **models/research** to your working directory. Just remember to change the import commands accordingly. 

After the modules have been downloaded/moved, you can use the code to in the **bounding_box.py** file to detect objects in your image. 

## Credit
Inspired by and based on https://rockyshikoku.medium.com/how-to-use-tensorflow-object-detection-api-with-the-colab-sample-notebooks-477707fadf1b

