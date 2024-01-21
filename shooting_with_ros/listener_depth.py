#!/usr/bin/env python
# -- coding: utf8 -- 
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
import cv2
import numpy as np
from cv_bridge import CvBridge


# Depth camera intrinsic parameters
K_depth = np.array([[1938.6278076171875, 0.0, 2053.2705078125],
                 [0.0, 1938.4910888671875, 1556.775146484375],
                 [0.0, 0.0, 1.0]])

D_depth = np.array([[0.5721315741539001, -2.5820364952087402, 0.0004875285958405584, 2.8991649742238224e-05, 1.471113681793213, 0.45420926809310913, -2.4220640659332275, 1.407278299331665]])


bridge = CvBridge()  

# Initialize global variables
key_value = ''
image_count = 0
day_dir = "./day/depth/"
night_dir = "./night/depth/"


def camera_callback(msg):   
    global key_value, image_count
    # receive image                                             
    thermal_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')        
    
    # Undistort rgb image
    depth_h, depth_w = thermal_image.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(K_depth, D_depth, (depth_w, depth_h), alpha = 0.01)
    undistorted_depth_image = cv2.undistort(thermal_image, K_depth, D_depth, None, new_matrix)
    
    # image show
    #cv2.imshow('Depth Image', depth_image)
    undistorted_depth_image = cv2.resize(undistorted_depth_image, (640, 480))
    cv2.imshow('Undistorted Depth Image', undistorted_depth_image)
    cv2.waitKey(1)
    
    # Image save
    if key_value == 'd':
        np.save(os.path.join(day_dir, "day_depth_%04i.npy" %image_count), undistorted_depth_image)
        # rospy.loginfo("Saved an image in Day directory")
    elif key_value == 'n':
        np.save(os.path.join(night_dir, "night_depth_%04i.npy" %image_count), undistorted_depth_image)
        # rospy.loginfo("Saved an image in Night directory")
    

def keys_callback(data):
    global key_value
    key_value = data.data
    rospy.loginfo("Received key: %s", key_value)

def count_callback(data):
    global image_count
    image_count = data.data
    rospy.loginfo("Received image count: %s", image_count)


def subscriber_thermal():
    rospy.init_node('camera_subscriber_depth', anonymous=True)    

    rospy.Subscriber("keys", String, keys_callback)
    rospy.Subscriber("image_count", Int32, count_callback)
    rospy.Subscriber('/depth_to_rgb/image_raw', Image, camera_callback)   
    
    rospy.spin() 

if __name__ == '__main__':
    subscriber_thermal()

