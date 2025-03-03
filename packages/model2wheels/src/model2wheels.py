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

from utils import crop_rgb_obs, resize_rgb_obs, apply_lane_detection_filter, process_img, grad_cam
from utils import steering_to_wheels_velocity_conversion, steering_to_wheel_velocities


class Model2WheelsNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(Model2WheelsNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)


        self._vehicle_name = os.environ['VEHICLE_NAME']

        print(f"Setting up model for {self._vehicle_name}")

        custom_objects = {
            "learning_rate": 3e-4,  # Set an appropriate value
            "lr_schedule": lambda x: 3e-4,  # Define a callable function
            "clip_range": lambda x: 0.2,  # Define a callable function
        }

        self.model = PPO.load('packages/model2wheels/src/models/duckietown_time_rand_new_reward_LATEST_model_trained_200000_steps',
                         custom_objects=custom_objects)
        print("Model loaded")
        print(f"cuda: {torch.cuda.is_available()}")



        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # create window
        self._window = "camera-reader2"

        #cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        #self.rate = rospy.Rate(8)
        self.counter = 0
        self.wheel_pub = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)
        self.camera_sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.camera_callback)


    def camera_callback(self, msg):
        # convert JPEG bytes to CV image
        print(f"Image: {self.counter} came in")
        image_full = self._bridge.compressed_imgmsg_to_cv2(msg)
        image_rgb, img_canny = process_img(image_full)
        #grad = grad_cam(self.model, image_rgb)
        #cv2.imshow(self._window, grad)
        #cv2.waitKey(1)
        # display frame
        if self.model:
            #cv2.imshow("IMG_RGB", image_rgb)
            #cv2.waitKey(1)
            action, _states = self.model.predict(image_rgb, deterministic=True)
            print(f"Action: {action}")
            left, right = steering_to_wheel_velocities(action)
            print(f"Left: {left}, right: {right}")
            self.publish_vels(left,right)
            print(f"Image: {self.counter} processed")
            #self.rate.sleep()
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
    node = Model2WheelsNode(node_name='model2wheels')
    # run node
    # keep the process from terminating
    rospy.spin()


