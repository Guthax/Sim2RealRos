#!/usr/bin/env python3

import os
import rospy
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

from utils import crop_rgb_obs, resize_rgb_obs, apply_lane_detection_filter, process_img, grad_cam
from utils import steering_to_wheels_velocity_conversion, steering_to_wheel_velocities
from utils_classes.dataloader import  DTSegmentationDataset
from fast_scnn import FastSCNN
class Seg2WheelsNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(Seg2WheelsNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)


        self._vehicle_name = os.environ['VEHICLE_NAME']

        print(f"Setting up model for {self._vehicle_name}")

        custom_objects = {
            "learning_rate": 3e-4,  # Set an appropriate value
            "lr_schedule": lambda x: 3e-4,  # Define a callable function
            "clip_range": lambda x: 0.2,  # Define a callable function
        }

        self.model = PPO.load('packages/seg2wheels/src/models/carla_rgb_80_height_cropped_better_sp_continued_small_lr_model_trained_200000_steps',
                         custom_objects=custom_objects)
        print("Model loaded for control")
        print(f"cuda: {torch.cuda.is_available()}")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device \: {self.device}")



        self._camera_topic = f"/{self._vehicle_name}/process_node/image/compressed"
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()


        #cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.counter = 0
        self.wheel_pub = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)
        self.camera_sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.camera_callback, queue_size=1)

    def camera_callback(self, msg):
        # convert JPEG bytes to CV image
        print(f"Image: {self.counter} came in")
        image_full = self._bridge.compressed_imgmsg_to_cv2(msg)
        cv2.imshow("img_seg", image_full)
        cv2.waitKey(1)
        """
        if self.model:
            #cv2.imshow("IMG_RGB", image_rgb)
            #cv2.waitKey(1)
            action, _states = self.model.predict(image_full, deterministic=True)
            print(f"Action: {action}")
            left, right = steering_to_wheel_velocities(action)
            print(f"Left: {left}, right: {right}")
            self.publish_vels(left,right)
            print(f"Image: {self.counter} processed")
            #self.rate.sleep()
        """

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


