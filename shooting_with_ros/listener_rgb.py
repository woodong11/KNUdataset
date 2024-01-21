#!/usr/bin/env python
# -- coding: utf8 -- 
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
import cv2
import numpy as np
from cv_bridge import CvBridge


# RGB camera intrinsic parameters
K_rgb = np.array([[1.95447930e+03, 0.00000000e+00, 2.03419789e+03],
                 [0.00000000e+00, 1.96211093e+03, 1.54255348e+03],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

D_rgb = np.array([[ 0.11531545, -0.08222561, -0.0072591 , -0.00503622,  0.01879449]])


bridge = CvBridge()  

# Initialize global variables
key_value = ''
image_count = 0
day_dir = "./day/rgb/"
night_dir = "./night/rgb/"


def camera_callback(msg):   
    global key_value, image_count
    # receive image                                            
    rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')        
    
    # Undistort rgb image
    rgb_h, rgb_w = rgb_image.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(K_rgb, D_rgb, (rgb_w, rgb_h), alpha = 0.01)
    undistorted_rgb_image = cv2.undistort(rgb_image, K_rgb, D_rgb, None, new_matrix)
    undistorted_rgb_image = cv2.resize(undistorted_rgb_image, (640, 480))
    # image show
    #cv2.imshow('RGB Image', rgb_image)
    #cv2.imshow('Undistorted RGB Image', undistorted_rgb_image)
    cv2.waitKey(1)
    
    # Image save
    if key_value == 'd':
        cv2.imwrite(os.path.join(day_dir, "day_rgb_%04i.png" %image_count), undistorted_rgb_image)
        # rospy.loginfo("Saved an image in Day directory")
    elif key_value == 'n':
        cv2.imwrite(os.path.join(night_dir, "night_rgb_%04i.png" %image_count), undistorted_rgb_image)
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
    rospy.init_node('camera_subscriber_thermal', anonymous=True)    

    rospy.Subscriber("keys", String, keys_callback)
    rospy.Subscriber("image_count", Int32, count_callback)
    rospy.Subscriber('/rgb/image_raw', Image, camera_callback)   
    
    rospy.spin() 

if __name__ == '__main__':
    subscriber_thermal()

