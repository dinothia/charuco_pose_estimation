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
from aruco_detection import ArucoDetector
from charuco_pose import CharucoPose

VERBOSE=False
SHOW_IMAGE=True

class image_subscriber:

    def __init__(self, camera_topic, marker_detector, pose_estimator, outfile_path):
        '''Initialize ros subscriber'''
        # subscribed Topic
        self.subscriber = rospy.Subscriber(camera_topic,
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print(f"subscribed to {camera_topic}")

        # marker detector and pose estimator
        self.marker_detector = marker_detector
        self.pose_estimator = pose_estimator

        self.current_time = 0
        self.previous_time = 0
        
        self.outfile_path = outfile_path

        self.timestamps, self.tvecs, self.rvecs = [], [], []
        self.current_timestamp = 0
        self.dt = 0.1

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print('received image of type: "%s"' % ros_data.format)

        ## Direct conversion to CV2 ####
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 

        ## Detect, filter markers and estimate pose
        corners, ids = self.marker_detector.get_corner_and_ids(frame)
        _, filt_corner = self.pose_estimator.filter_ids(ids, corners)
        rvec, tvec = self.pose_estimator.estimate_pose(filt_corner)

        ## Append pose to list
        self.timestamps.append(self.current_timestamp)
        self.tvecs.append(tvec)
        self.rvecs.append(rvec)

        if SHOW_IMAGE:
            marker_frame = self.marker_detector.draw_markers(frame, corners, ids)
            cv2.imshow('cv_img', marker_frame)
            cv2.waitKey(2)

        if VERBOSE:
            print(f"Hz: {self.get_frequency()}")

        self.current_timestamp += self.dt

    def get_frequency(self):
        self.current_time = rospy.get_time()
        frequency = 1.0 / (self.current_time - self.previous_time)
        self.previous_time = self.current_time\
                    
        
        return frequency

    def save_to_file(self):
        with open(self.outfile_path, "w") as file:
            for t, tvec, rvec in zip(self.timestamps, self.tvecs, self.rvecs):
                file.write(f"{round(t, 2)}, {tvec[0][0][0]}, {tvec[0][0][1]}, {tvec[0][0][2]}, {rvec[0][0][0]}, {rvec[0][0][1]}, {rvec[0][0][2]}\n")


def main(args):
    # Get commandline arguments
    camera_topic = args[1]
    camera_path_path = args[2]
    outfile_path = args[3]

    marker_detector = ArucoDetector()

    # Pose estimator large marker
    x, y = 3, 3
    square_size = 0.300  # m
    marker_size = 0.225  # m 
    marker_id = 0
    pose_estimator = CharucoPose(camera_path_path, x, y, square_size, marker_size, marker_id)

    '''Initializes and cleanup ros node'''
    ic = image_subscriber(camera_topic, marker_detector, pose_estimator, outfile_path)
    rospy.init_node('image_subscriber', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature marker_detector module")
    cv2.destroyAllWindows()

    ic.save_to_file()


if __name__ == '__main__':
    main(sys.argv)