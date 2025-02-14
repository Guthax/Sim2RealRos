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
import sched, time

from utils import convert_steering_to_wheel_vels


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

        self.model = PPO.load('packages/model2wheels/src/models/carla_only_rgb_steering_model_trained_200000_steps.zip',
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
        self.rate = rospy.Rate(1000)

        self.camera_sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.camera_callback)
        self.wheel_pub = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

        self.ttl = 0
        #self.my_scheduler = sched.scheduler(time.time, time.sleep)
        #self.my_scheduler.enter(60, 1, self.update_ttl, (self.my_scheduler,))
        #self.my_scheduler.run()


        #self.publish_vels(0.2, 0.2)


    def camera_callback(self, msg):
        # convert JPEG bytes to CV image

        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        image = cv2.resize(image, (160, 80))
        #cv2.imshow(self._window, image)
        #cv2.waitKey(1)

        print(image.shape)
        dict = {
            "rgb_camera": image
        }

        # display frame
        if self.model:
            action, _states = self.model.predict(dict, deterministic=True)
            left, right = convert_steering_to_wheel_vels(action[0])
            print(f"Left: {left}, right: {right}")
            self.publish_vels(left,right)
            self.rate.sleep()




    def publish_vels(self, vel_1: float, vel_2: float):
        print(f"Publishing vels: {vel_1}, {vel_2}")
        message = WheelsCmdStamped(vel_left=vel_1, vel_right=vel_2)
        self.wheel_pub.publish(message)
        #self.rate.sleep()

    def update_ttl(self, scheduler):
        # schedule the next call first
        scheduler.enter(60, 1, self.update_ttl, (scheduler,))
        self.ttl += 1
        print("Setting TTL to: ", self.ttl)
        # then do your stuff

    def on_shutdown(self):
        print("Shutdown")
        print(f"TTL: {self.ttl}")
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self.wheel_pub.publish(stop)

if __name__ == '__main__':
    # create the node
    node = Model2WheelsNode(node_name='model2wheels')
    # run node
    # keep the process from terminating
    rospy.spin()
