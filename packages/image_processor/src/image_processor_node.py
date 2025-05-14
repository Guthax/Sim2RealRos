#!/usr/bin/env python3
import os
import rospy
import time
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from torchvision.transforms import transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from utils_classes.dataloader_basic import DTSegmentationDataset
from fast_scnn import FastSCNN

cudnn.benchmark = True


latest_compressed_image = None
def image_callback(msg):
    global latest_compressed_image
    latest_compressed_image = msg




class ImageProcessor(DTROS):
    def __init__(self, node_name):
        super(ImageProcessor, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        print(f"cuda: {torch.cuda.is_available()}")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device \: {self.device}")


        # Load model
        num_classes = len(DTSegmentationDataset.SEGM_LABELS)
        self.model_segmentation = FastSCNN(num_classes=num_classes)
        self.model_segmentation.load_state_dict(
            torch.load('packages/image_processor/src/models/fast_scnn_best.pth', map_location=self.device)
        )
        self.model_segmentation.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((480, 640)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Model loaded for segmentation inference")

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._process_topic = f"/{self._vehicle_name}/process_node/image/one_hot"

        self._bridge = CvBridge()
        #self.process_pub = rospy.Publisher(self._process_topic, CompressedImage, queue_size=1)
        self.process_pub = rospy.Publisher(self._process_topic, Float32MultiArray, queue_size=1)

        # Set subscriber with queue_size=1 to avoid backlog
        self.camera_sub = rospy.Subscriber(
            self._camera_topic, CompressedImage, image_callback, queue_size=1)

        print("Completed setup")
        while not rospy.is_shutdown():
            if latest_compressed_image is not None:
                self.process(latest_compressed_image)


    def process(self, img_compressed):
        # Only process the newest incoming image
        try:
            cv_image = self._bridge.compressed_imgmsg_to_cv2(img_compressed)
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
        pred = pred / 3
        pred = cv2.resize(pred, (160, 120), interpolation=cv2.INTER_NEAREST)
        #one_hot = DTSegmentationDataset.label_img_to_one_hot(pred)
        #one_hot = self.resize_one_hot(one_hot)

        msg = Float32MultiArray()
        msg.layout.dim.append(MultiArrayDimension(label="height", size=pred.shape[0], stride=pred.size))
        msg.layout.dim.append(MultiArrayDimension(label="width", size=pred.shape[1], stride=pred.shape[1]))
        msg.data = pred.flatten().tolist()


        # Publish the processed image
        #_, image_jpg = cv2.imencode('.jpg', pred_rgb)
        #msg = CompressedImage()
        #msg.header.stamp = rospy.Time.now()
        #msg.format = "jpeg"
        #msg.data = np.array(image_jpg).tobytes()

        self.process_pub.publish(msg)
        print("Published processed image")
        return DTSegmentationDataset.label_img_to_rgb(pred)


    def resize_one_hot(self, one_hot: np.ndarray, size=(120, 160)) -> np.ndarray:
        # one_hot shape: [C, H, W]
        one_hot_tensor = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)  # [1, C, H, W]

        # Get label map (class index per pixel)
        label_map = torch.argmax(one_hot_tensor, dim=1, keepdim=True)  # [1, 1, H, W]

        # Resize using nearest neighbor (preserves class labels)
        resized_labels = F.interpolate(label_map.float(), size=size, mode='nearest')  # [1, 1, H_new, W_new]

        # Back to one-hot
        resized_labels = resized_labels.squeeze(0).squeeze(0).long()  # [H_new, W_new]
        num_classes = one_hot.shape[0]
        one_hot_resized = torch.nn.functional.one_hot(resized_labels, num_classes=num_classes)  # [H, W, C]
        one_hot_resized = one_hot_resized.permute(2, 0, 1).numpy()  # [C, H, W]

        return one_hot_resized.astype(np.int8)

if __name__ == '__main__':
    node = ImageProcessor(node_name='image_processor')
    rospy.spin()
