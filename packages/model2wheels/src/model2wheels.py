#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge

class Model2WheelNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(Model2WheelNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # create window
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.camera_sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.camera_callback)
        self.wheel_pub = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

    def camera_callback(self, msg):
        print("Camera Callback")
        # convert JPEG bytes to CV image
        #image = self._bridge.compressed_imgmsg_to_cv2(msg)
        # display frame
        #cv2.imshow(self._window, image)
        #cv2.waitKey(1)
        #self.publish_vels(0.2,0.2)


    def publish_vels(vel_1: float, vel_2: float):
        print("Publish vels")
        #message = WheelsCmdStamped(vel_left=self.vel_1, vel_right=vel_2)
        #self.wheel_pub.publish(message)

    def on_shutdown(self):
        print("Shutdown")
        #stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        #self.wheel_pub.publish(stop)
