#!/usr/bin/env python3
import os
import rospy
import time
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from torchvision.transforms import transforms
from threading import Lock

from utils_classes.dataloader import DTSegmentationDataset
from fast_scnn import FastSCNN
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


class ImageProcessor(DTROS):
    def __init__(self, node_name):
        super(ImageProcessor, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.device = "cpu"
        print(f"Device: {self.device}")

        num_classes = len(DTSegmentationDataset.SEGM_LABELS)
        self.model_segmentation = FastSCNN(num_classes=num_classes)
        self.model_segmentation.load_state_dict(
            torch.load('packages/image_processor/src/models/fast_scnn_epoch_50.pth'))
        self.model_segmentation.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 480)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Model loaded for segmentation inference")

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._process_topic = f"/{self._vehicle_name}/process_node/image/compressed"    

        self._bridge = CvBridge()
        self.process_pub = rospy.Publisher(self._process_topic, CompressedImage, queue_size=1)
        self.camera_sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.camera_callback)

        self.img_buff = None
        self.lock = Lock()

        print("Completed setup")

    def camera_callback(self, msg):
        print("Image came in")
        self.img_buff = self._bridge.compressed_imgmsg_to_cv2(msg)
        start_time = time.time()
        self.process_image(self.img_buff)
        end_time = time.time()
        print(f"Elapsed: {end_time - start_time}")


    def run(self):
        while not rospy.is_shutdown():
            print("In loop")
            if self.img_buff is not None:
                print("Copying buffer")
                image_full = self.img_buff.copy()

                start_time = Time.time()
                self.process_image(image_full)
                end_time = Time.time()
                print(f"Elapsed: {end_time - start_time}")
    def process_image(self, image_full):
        print("Processing latest image")
        img = self.transform(image_full).unsqueeze(0).to(self.device, non_blocking=True)

        with torch.no_grad():
            output = self.model_segmentation(img)[0]

        pred = torch.argmax(output, dim=1).cpu().squeeze().numpy()
        pred = DTSegmentationDataset.label_img_to_rgb(pred)
        _, image_jpg = cv2.imencode('.jpg', pred)
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(image_jpg).tobytes()

        self.process_pub.publish(msg)
        print("Published processed image")


if __name__ == '__main__':
    node = ImageProcessor(node_name='image_processor')
    rospy.spin()
