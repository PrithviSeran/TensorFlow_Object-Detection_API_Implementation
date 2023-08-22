from PIL import Image
import matplotlib 
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import tensorflow as tf
import numpy as np 
from six import BytesIO 
from PIL import Image
import tensorflow as tf  
from object_detection.utils import label_map_util 
from object_detection.utils import config_util 
from object_detection.utils import visualization_utils as viz_utils 
from object_detection.builders import model_builder 

#opens images from the order specified in a csv file (see README file)
#only use if your are loading a dataset of images to train the model
def open_images(image_path):
  ordered_images = []
  with open(image_path, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader: 
      ordered_images.append(np.asarray(Image.open(f'images/{row[0]}')))
  
  return ordered_images

#getting the ordered images for training/testing
ordered_images = open_images("data/train_labels.csv")

#path of the .config file
pipeline_config = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config" 

#path of the pre-trained model's checkpoint
model_dir = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint"

#getting the configuration of the model from the .config file
configs = config_util.get_configs_from_pipeline_file(pipeline_config) 

#getting the configuration of the model
model_config = configs['model']

#bulding the model based on the configuration
detection_model = model_builder.build(model_config=model_config, is_training=True)

#loading the checkpoint (loading the weights)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model) 
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

#if in eval_config it specificies keypoint edges, returns (star, end) of the keypoints. (Used for landmark detections)
def get_keypoint_tuples(eval_config):
   """Return a tuple list of keypoint edges from the eval config.
    Args:
     eval_config: an eval config containing the keypoint edges
    Returns:
     a list of edge tuples, each in the format (start, end)
   """
   tuple_list = []
   kp_list = eval_config.keypoint_edge
   for edge in kp_list:
     tuple_list.append((edge.start, edge.end))
   return tuple_list

#returns the dimensions and coordinates of the bounding boxes produced by the model
def get_model_detection_function(model):

    """Get a tf.function for detection."""
    @tf.function
    def detect_fn(image):
        """Detect objects in image."""
        image, shapes = model.preprocess(image)

        prediction_dict = model.predict(image, shapes)

        detections = model.postprocess(prediction_dict, shapes) 

        return detections, prediction_dict, tf.reshape(shapes, [-1])
    return detect_fn

#creating the detections object 
detect_fn = get_model_detection_function(detection_model)

#getting all the classes for the model
label_map_path = 'object_detection/data/mscoco_label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path) 
categories = label_map_util.convert_label_map_to_categories(
     label_map,
     max_num_classes=label_map_util.get_max_label_map_index(label_map),
     use_display_name=True) 

#formatting all the classes
category_index = label_map_util.create_category_index(categories) 

#The image you want bounding boxes over
image_np = "OPEN IMAGE AS AN ARRAY USING PIL"
 
#converting the image into a tensor
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32) 

#getting the detections (bounding boxes)
detections, predictions_dict, shapes = detect_fn(input_tensor)  


label_id_offset = 1 
image_np_with_detections = image_np.copy()  
keypoints, keypoint_scores = None, None 

#if there are keypoints that need to be detected, set them to a value. if not they will be set as None. 
if 'detection_keypoints' in detections:
   keypoints = detections['detection_keypoints'][0].numpy()
   keypoint_scores = detections['detection_keypoint_scores'][0].numpy()


#overlaying the bounding over the image. This function groups all the bounding boxes in the same area as one
viz_utils.visualize_boxes_and_labels_on_image_array(
       image_np_with_detections,
       detections['detection_boxes'][0].numpy(),
       (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
       detections['detection_scores'][0].numpy(),
       category_index,
       use_normalized_coordinates=True,
       max_boxes_to_draw=200,
       min_score_thresh=.50,
       agnostic_mode=False,
       keypoints=keypoints,
       keypoint_scores=keypoint_scores,
       keypoint_edges=get_keypoint_tuples(configs['eval_config']))


#opening and showing the image
img = Image.fromarray(image_np_with_detections, "RGB")
img.show()
