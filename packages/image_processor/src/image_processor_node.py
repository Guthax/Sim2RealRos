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
import torch.backends.cudnn as cudnn

from utils_classes.dataloader import DTSegmentationDataset
from fast_scnn import FastSCNN

cudnn.benchmark = True


class ImageProcessor(DTROS):
    def __init__(self, node_name):
        super(ImageProcessor, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.device = "cpu"
        print(f"Device: {self.device}")

        # Load model
        num_classes = len(DTSegmentationDataset.SEGM_LABELS)
        self.model_segmentation = FastSCNN(num_classes=num_classes)
        self.model_segmentation.load_state_dict(
            torch.load('packages/image_processor/src/models/fast_scnn_epoch_50.pth', map_location=self.device)
        )
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

        # Set subscriber with queue_size=1 to avoid backlog
        self.camera_sub = rospy.Subscriber(
            self._camera_topic, CompressedImage, self.camera_callback, queue_size=1, buff_size=2**24
        )


        print("Completed setup")

    def camera_callback(self, msg):
        # Only process the newest incoming image
        try:
            cv_image = self._bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logwarn(f"Could not decode image: {e}")
            return

        start_time = time.time()
        img = self.process_image(cv_image)
        end_time = time.time()
        rospy.loginfo(f"Image processed in {end_time - start_time:.3f} seconds")

    def process_image(self, image_full):
        print("Processing latest image")
        img = self.transform(image_full).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model_segmentation(img)[0]

        pred = torch.argmax(output, dim=1).cpu().squeeze().numpy()
        pred_rgb = DTSegmentationDataset.label_img_to_rgb(pred)

        # Optional: Display result


        # Publish the processed image
        _, image_jpg = cv2.imencode('.jpg', pred_rgb)
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(image_jpg).tobytes()

        self.process_pub.publish(msg)
        print("Published processed image")
        return pred_rgb


if __name__ == '__main__':
    node = ImageProcessor(node_name='image_processor')
    rospy.spin()
