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
from plot_pose import (
    save_pose, 
    read_pose,
    plot_orientation, 
    plot_translation, 
    show_all
)

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

        # outfile path for pose
        self.outfile_path = outfile_path

        # pose timestamps
        self.current_time = 0
        self.previous_time = 0

        # initialize pose lists
        self.timestamps, self.tvecs, self.rvecs = [], [], []
        self.camera_timestamp = 0
        self.camera_dt = 0.1  # sampling time for camera 10 Hz

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
        self.timestamps.append(self.camera_timestamp)
        self.tvecs.append(tvec)
        self.rvecs.append(rvec)

        if SHOW_IMAGE:
            marker_frame = self.marker_detector.draw_markers(frame, corners, ids)
            frame = self.pose_estimator.draw_marker_axis(frame, rvec, tvec, 0.2)
            
            cv2.imshow('cv_img', marker_frame)
            cv2.waitKey(2)

        print(f"t: {round(self.camera_timestamp, 2)} Hz: {self.get_frequency()}")
        self.camera_timestamp += self.camera_dt

    def get_frequency(self):
        self.current_time = rospy.get_time()
        frequency = 1.0 / (self.current_time - self.previous_time)
        self.previous_time = self.current_time
        return frequency

    def save_to_file(self):
        save_pose(self.outfile_path, self.timestamps, self.tvecs, self.rvecs)
        print(f"Saving keyframe trajectory to {self.outfile_path} ...")

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

    # Initializes and cleanup ros node
    ic = image_subscriber(camera_topic, marker_detector, pose_estimator, outfile_path)
    rospy.init_node('image_subscriber', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature marker_detector module")
    cv2.destroyAllWindows()

    # Save pose to file when ctrl+C/exit
    ic.save_to_file()

    # Read pose
    t, tvecs, eulers = read_pose(outfile_path)
    eulers_deg = 180 * eulers / np.pi

    # Plot pose
    plot_translation(t, tvecs)
    plot_orientation(t, eulers_deg)
    show_all()


if __name__ == '__main__':
    main(sys.argv)
