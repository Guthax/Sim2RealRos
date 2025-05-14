#!/usr/bin/env python3

import os
import rospy
import sys
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from stable_baselines3 import PPO
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import sched, time
from torchvision.transforms import transforms
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from utils import steering_to_wheels_velocity_conversion, steering_to_wheel_velocities
from fast_scnn import FastSCNN
np.set_printoptions(threshold=sys.maxsize)
latest_processed_image = None
def processed_image_callback(msg):
    global latest_processed_image
    latest_processed_image = msg  # Just store the latest message


class Seg2WheelsNode(DTROS):
    def __init__(self, node_name):
        # initialize Seg2WheelsNode DTROS parent class
        super(Seg2WheelsNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)


        self._vehicle_name = os.environ['VEHICLE_NAME']

        print(f"Setting up model for {self._vehicle_name}")

        custom_objects = {
            "learning_rate": 3e-4,  # Set an appropriate value
            "lr_schedule": lambda x: 3e-4,  # Define a callable function
            "clip_range": lambda x: 0.2,  # Define a callable function
        }

        self.model = PPO.load('packages/seg2wheels/src/models/carla_seg_256_no_crop_model_trained_600000_steps',
                         custom_objects=custom_objects)
        print("Model loaded for control")
        print(f"cuda: {torch.cuda.is_available()}")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device \: {self.device}")

        self.last_action = 0.0
        self._one_hot_topic = f"/{self._vehicle_name}/process_node/image/one_hot"
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()


        #cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.counter = 0
        self.wheel_pub = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)
        self.camera_sub = rospy.Subscriber(self._one_hot_topic, Float32MultiArray, processed_image_callback, queue_size=1)

        while not rospy.is_shutdown():
            if latest_processed_image is not None:
                self.process(latest_processed_image)

    def process(self, msg):
        # convert JPEG bytes to CV image
        print("Msg came in")
        dims = msg.layout.dim
        H, W = dims[0].size, dims[1].size
        obs_img = np.expand_dims(np.array(msg.data, dtype=np.float32).reshape(H, W), axis=0)
        #channel_max = one_hot.max(axis=(0, 1))
        obs = {
            "camera_seg": obs_img,
            "vehicle_dynamics": [self.last_action],
        }
        if self.model:

            action, _states = self.model.predict(obs, deterministic=True)
            print(f"Action: {action}")
            left, right = steering_to_wheel_velocities(action)
            self.last_action = action[0]
            print(f"Left: {left}, right: {right}")
            self.publish_vels(left, right)
            print(f"Image: {self.counter} processed")

        self.counter += 1

    def publish_vels(self, vel_1: float, vel_2: float):
        print(f"Publishing vels: {vel_1}, {vel_2}")
        message = WheelsCmdStamped(vel_left=vel_1, vel_right=vel_2)
        self.wheel_pub.publish(message)
        #self.rate.sleep()

    def on_shutdown(self):
        print("Shutdown")
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        #self.rate.sleep()
        self.wheel_pub.publish(stop)
        self.wheel_pub.publish(stop)
        self.wheel_pub.publish(stop)



if __name__ == '__main__':
    # create the node
    node = Seg2WheelsNode(node_name='seg2wheels')
    # run node
    # keep the process from terminating
    rospy.spin()


