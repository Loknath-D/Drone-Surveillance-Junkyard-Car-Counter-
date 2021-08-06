import dlib
import cv2
import os
import numpy as np
import time
from lxml import etree

images = [];
bbox = [];

'''Extract the training images and respective bounding boxes from XML files within Training Data Directory'''
for file in os.listdir('./Drone_Surveillance_Counter/Training Data'):
    if file.endswith(".jpg"):
        images.append(cv2.imread('./Drone_Surveillance_Counter/Training Data/' + file));
    if file.endswith(".xml"):
        with open('./Drone_Surveillance_Counter/Training Data/' + file) as xml:
            root = etree.XML(xml.read());
        bbox.append((int(root[6][4][0].text), int(root[6][4][1].text), int(root[6][4][2].text), int(root[6][4][3].text)));

'''Convert the bounding boxes to dlib format'''
dlib_bbox = [[dlib.rectangle(left = b[0], top = b[1], right = b[2], bottom = b[3])] for b in bbox];

'''Instantiate HOG and SVM Object Detector Options'''
options = dlib.simple_object_detector_training_options();

'''Disable Symmetrical Detections in Object Detector Options'''
options.add_left_right_image_flips = False;

'''Initialize C value'''
'''A bigger C encourages the model to better fit the training data, it can lead to overfitting.'''
options.C = 7.605;

'''Note the start time of training'''
st = time.time();

'''Start training'''
detector = dlib.train_simple_object_detector(images, dlib_bbox, options);

'''Show the total time taken to train'''
print('Training Completed (C = 7.605)!!!');
print('Total Time Taken: {:.2f} seconds'.format(time.time() - st));

'''Show the testing metrics with 20% of data'''
print("Testing Metrics: {}".format(dlib.test_simple_object_detector(images[80:], dlib_bbox[80:], detector)));

'''Save the detector in the directory'''
detector_file = 'Car_Detector_1 (C = 7.605).svm';
detector.save(detector_file);
print('Trained Detector saved in the working directory.');


