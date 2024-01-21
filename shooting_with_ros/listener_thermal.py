#!/usr/bin/env python
# -- coding: utf8 -- 
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
import cv2
import numpy as np
from cv_bridge import CvBridge

bridge = CvBridge()  


# Initialize global variables
key_value = ''
image_count = 0
day_dir = "./day/thermal/"
night_dir = "./night/thermal/"

# Thermal camera parameters

K_thermal = np.array([[412.30837785,  0.        , 315.12693405],
                      [0.        , 412.31541992, 244.68299369],
                      [0.        ,   0.        ,   1.        ]])
D_thermal = np.array([[-4.02590909e-01,  2.31041396e-01, -4.89959646e-05, -2.48363343e-04, -8.06806958e-02]])

H = np.array([[ 7.44488229e-01,  7.60187945e-03,  6.92041593e+01],
              [-3.79541197e-02,  7.67188061e-01,  4.06300075e+01],
              [-8.11637699e-05,  4.07947462e-05,  1.00000000e+00]])



def camera_callback(msg):   
    global key_value, image_count
    # receive image                                             
    thermal_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')        
    
    # Undistort thermal image
    thermal_h, thermal_w = thermal_image.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(K_thermal, D_thermal, (thermal_w, thermal_h), alpha = 0.01)
    undistorted_thermal_image = cv2.undistort(thermal_image, K_thermal, D_thermal, None, new_matrix)
    
    # Apply Homograpy
    #thermal_gray = cv2.cvtColor(undistorted_thermal_image, cv2.COLOR_BGR2GRAY)
    result = cv2.warpPerspective(undistorted_thermal_image, H, (640, 480))
    moved_result = cv2.copyMakeBorder(result, 8, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # image show
    #cv2.imshow('Thermal Image', thermal_image)
    #cv2.imshow('Undistorted thermal Image', undistorted_thermal_image)
    cv2.imshow('Homograpy', moved_result)
    cv2.waitKey(1)
    
    # Image save
    if key_value == 'd':
        cv2.imwrite(os.path.join(day_dir, "day_thermal_%04i.png" %image_count), moved_result)
        # rospy.loginfo("Saved an image in Day directory")
    elif key_value == 'n':
        cv2.imwrite(os.path.join(night_dir, "night_thermal_%04i.png" %image_count), moved_result)
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
    rospy.Subscriber('/flir_boson/image_raw', Image, camera_callback)   
    
    rospy.spin() 

if __name__ == '__main__':
    subscriber_thermal()
