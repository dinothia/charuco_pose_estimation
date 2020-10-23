#!/usr/bin/env python
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""
__author__ =  'Simon Haller <simon.haller at uibk.ac.at>'
__version__=  '0.1'
__license__ = 'BSD'

# NOTE: Based by: http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

# Marker libraries
from aruco_detection import aruco_detector

VERBOSE=False
SHOW_IMAGE=True

class image_subscriber:

    def __init__(self, camera_topic, detector):
        '''Initialize ros subscriber'''
        # subscribed Topic
        self.subscriber = rospy.Subscriber(camera_topic,
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print(f"subscribed to {camera_topic}")

        # detector
        self.detector = detector

        self.current_time = 0
        self.previous_time = 0

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print('received image of type: "%s"' % ros_data.format)

        #### direct conversion to CV2 ####
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
        corners, ids = self.detector.get_corner_and_ids(frame)
                
        if SHOW_IMAGE:
            marker_frame = self.detector.draw_markers(frame, corners, ids)

            cv2.imshow('cv_img', marker_frame)
            cv2.waitKey(2)

        print(f"Hz: {self.get_frequency()}")

    def get_frequency(self):
        self.current_time = rospy.get_time()
        frequency = 1.0 / (self.current_time - self.previous_time)
        self.previous_time = self.current_time
        
        return frequency

def main(args):
    camera_topic = args[1]
    detector = aruco_detector()

    '''Initializes and cleanup ros node'''
    ic = image_subscriber(camera_topic, detector)
    rospy.init_node('image_subscriber', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)