#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag."""

import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag. """
    # parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    # parser.add_argument("bag_file", help="Input ROS bag.")
    # parser.add_argument("output_dir", help="Output directory.", default="./")
    # parser.add_argument("image_topic", help="Image topic.")

    # args = parser.parse_args()

    # print "Extract images from %s on topic %s into %s" % (args.bag_file,
    #                                                       args.image_topic, args.output_dir)

    bag_foler = "/home/woodong/ws/rosbag/0629"
    output_dir = "/home/woodong/ws/rosbag/0629"
    # image_topic = "/cam_1/color/image_raw"
    iter = 0

    import chardet


    for _, bag_file in enumerate(sorted(os.listdir(bag_foler))):
        iter += 1
        print(bag_file)
        bag = rosbag.Bag(os.path.join(bag_foler, bag_file), "r")
        bridge = CvBridge()

        # thermal
        count = 0
        for topic, msg, t in bag.read_messages(topics=["/flir_boson/image_raw"]):
            if count == 0:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                cv2.imwrite(os.path.join(output_dir, "thermal_%02i.png" %iter), cv_img)
                # print "Wrote image %i" % count
            count += 1

        # color
        count = 0  
        for topic, msg, t in bag.read_messages(topics=["/rgb/image_raw"]):
            if count == 0:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # BGR to RGB
                cv2.imwrite(os.path.join(output_dir, "color_%02i.png" %iter), cv_img_rgb)
                print("Wrote image %i" % count)

            count += 1

        # depth
        count = 0  
        for topic, msg, t in bag.read_messages(topics=["/depth_to_rgb/image_raw"]):
            if count == 0:
                cv_img = np.array(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
                np.save(os.path.join(output_dir,'depth_%02i.npy' %iter), cv_img)
                cv2.imwrite(os.path.join(output_dir, "depth_%02i.png" %iter), cv_img)     # png로 저장할때 
                print ("Wrote image %i" % count)

            count += 1
            
        bag.close()


    return

if __name__ == '__main__':
    main()