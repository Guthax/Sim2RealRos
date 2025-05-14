#!/usr/bin/env python3
from pynput import keyboard
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


latest_compressed_image = None

def compressed_image_callback(msg):
    global latest_compressed_image
    latest_compressed_image = msg  # Just store the latest message

class RGB2WheelsNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(RGB2WheelsNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)


        self._vehicle_name = os.environ['VEHICLE_NAME']

        print(f"Setting up model for {self._vehicle_name}")

        custom_objects = {
            "learning_rate": 3e-4,  # Set an appropriate value
            "lr_schedule": lambda x: 3e-4,  # Define a callable function
            "clip_range": lambda x: 0.2,  # Define a callable function
        }

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"

        self.wheel_pub = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)
        self.camera_sub = rospy.Subscriber(self._camera_topic, CompressedImage, compressed_image_callback, queue_size=1)

        self.model = PPO.load('packages/rgb2wheels/src/models/duckie_rgb_256_baseline_crop_lm_fix_model_trained_200000_steps',
                         custom_objects=custom_objects)
        print("Model loaded for control")
        print(f"cuda: {torch.cuda.is_available()}")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device \: {self.device}")

        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'mp4v' for .mp4 output
        #self.video_writer = cv2.VideoWriter("duckiebot_video.mp4", fourcc, 20, (160, 120))

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        #listener = keyboard.Listener(on_press=self.on_press)
        #listener.start()

        #cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        #self.rate = rospy.Rate(50)
        self.counter = 0
        self.last_action = 0
        while not rospy.is_shutdown():
            if latest_compressed_image is not None:
                self.process(latest_compressed_image)

    def process (self, msg):
        # convert JPEG bytes to CV image
        print(f"Image: {self.counter} came in")
        image_full = self._bridge.compressed_imgmsg_to_cv2(msg)
        #self.video_writer.write(image_resized)  # Write the frame to video

        image_processed, _ = process_img(image_full)
        obs = {
            "camera_rgb": image_processed,
            "vehicle_dynamics": [self.last_action],
        }
        if self.model:
            #cv2.imshow("IMG_RGB", image_rgb)
            #cv2.waitKey(1)
            action, _states = self.model.predict(obs, deterministic=True)
            print(f"Action: {action}")
            left, right = steering_to_wheel_velocities(action)
            self.last_action = action[0]
            print(f"Left: {left}, right: {right}")
            self.publish_vels(left,right)
            print(f"Image: {self.counter} processed")

        #self.rate.sleep()
        self.counter += 1

    def on_press(self,key):
        try:
            print("r is pressed, saving recording")
            if key.char == 'r':
                try:
                    self.video_writer.release()
                    print("recording saved")
                except:
                    print("Trying to release but didnt work")
            elif key.char == 's':
                rospy.loginfo("S pressed")
        except AttributeError:
            pass

    def publish_vels(self, vel_1: float, vel_2: float):
        print(f"Publishing vels: {vel_1}, {vel_2}")
        message = WheelsCmdStamped(vel_left=vel_1, vel_right=vel_2)
        self.wheel_pub.publish(message)
        #self.rate.sleep()

    def on_shutdown(self):
        print("Shutdown")
        print("Written")
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        #self.rate.sleep()
        if self.wheel_pub:
            self.wheel_pub.publish(stop)
            self.wheel_pub.publish(stop)
            self.wheel_pub.publish(stop)

if __name__ == '__main__':
    # create the node
    node = RGB2WheelsNode(node_name='seg2wheels')
    # run node
    # keep the process from terminating
    rospy.spin()


