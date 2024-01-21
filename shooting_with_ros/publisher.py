#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Int32
import sys
import select
import tty

image_count = 54

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    return key

def keys_publisher(image_count):
    rospy.loginfo("image_count: %d", image_count)
    pub_key = rospy.Publisher('keys', String, queue_size=1)
    pub_count = rospy.Publisher('image_count', Int32, queue_size=1)

    rospy.init_node('keys_publisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        key = getKey()
        if key == 'd' or key == 'n':
            rospy.loginfo("Key pressed: %s", key)
            rospy.loginfo("Image count: %s ", image_count)
            rospy.loginfo("\n")
            pub_key.publish(key)
            pub_count.publish(image_count)
            image_count += 1
        # if 'q' is pressed, the node will shutdown
        elif key == 'q':  
            rospy.loginfo("Shutting down...")
            rospy.signal_shutdown("Quit requested by user")
        rate.sleep()

if __name__ == '__main__':
    try:
        print("image count:",image_count)
        keys_publisher(image_count)
    except rospy.ROSInterruptException:
        pass
