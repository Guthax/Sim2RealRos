from typing import Union, List
import numpy as np
import cv2
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor

def steering_to_wheels_velocity_conversion(steering):
    # Ensure the steering angle is within the valid range
    steering_angle = max(-1, min(1, steering))

    # Map the steering angle to wheel velocities
    left_wheel_velocity = 0.25 * (1 + steering_angle)
    right_wheel_velocity = 0.25 * (1 - steering_angle)

    return  1.2 * left_wheel_velocity, 1.2 * right_wheel_velocity

def steering_to_wheel_velocities(steering):
    """
    Convert a steering variable into differential drive wheel velocities.

    Parameters:
        steering (float): A value between -1 (full left) and 1 (full right).

    Returns:
        tuple: (left_wheel_velocity, right_wheel_velocity)
    """
    # Ensure steering is within valid bounds
    steering = max(-1, min(1, 2* steering))

    # Map steering to wheel velocities
    #left_wheel_velocity = (1 + steering) / 2
    #right_wheel_velocity = (1 - steering) / 2
    left_wheel_velocity = 0.25 * (2 + steering)
    right_wheel_velocity = 0.25 * (2 - steering)

    return left_wheel_velocity, right_wheel_velocity

def process_img(image_rgb_full):
    img_rgb_resized = resize_rgb_obs(image_rgb_full, 160, 120)

    img_rgb_cropped = crop_rgb_obs(img_rgb_resized, 40, 120)
    #image_canny_cropped = apply_lane_detection_filter(img_rgb_cropped)

    return img_rgb_cropped, None

def resize_rgb_obs(obs, dst_w, dst_h):
    return cv2.resize(obs, (dst_w, dst_h))

def crop_rgb_obs(obs, start_h = None, end_h = None, start_w=None, end_w=None):
    img = obs
    if start_h:
        img = obs[start_h:end_h, :, :]

    if start_w:
        img = obs[:, start_w, end_w, :]

    return img

def apply_lane_detection_filter(input_img):

    def detect_lanes(img):
        # Load the image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 100, 300)

        # Apply region of interest mask
        # roi = self.region_of_interest(edges)

        return edges

    return detect_lanes(input_img)



def grad_cam(model, obs):

    obs = np.transpose(obs, (2,0,1))
    obs = np.expand_dims(obs, axis=0)
    policy_net = model.policy
    last_cnn_layer = policy_net.features_extractor.cnn[0]

    activations = {}
    gradients = {}

    # Hook for forward pass (store activations)
    def forward_hook(module, input, output):
        print(output)
        activations["features"] = output

    # Hook for backward pass (store gradients)
    def backward_hook(module, grad_input, grad_output):
        gradients["features"] = grad_output[0]


    forward_handle = last_cnn_layer.register_forward_hook(forward_hook)
    backward_handle = last_cnn_layer.register_backward_hook(backward_hook)

    tensor = obs_as_tensor(obs, device='cuda' if torch.cuda.is_available() else 'cpu')
    #tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = preprocess_obs(tensor, model.observation_space)
    #tensor = tensor.permute(3, 1, 2)

    with torch.set_grad_enabled(True):
        gaussian = model.policy.get_distribution(tensor)
        action_distribution = gaussian.distribution
        action_sample = action_distribution.sample()  # Sample an action
        log_prob = action_distribution.log_prob(action_sample)  # Compute log probability

        # Compute gradients for Grad-CAM
        log_prob.sum().backward()

    activations = activations["features"].detach()
    gradients = gradients["features"].detach()

    # Compute Grad-CAM heatmap
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Global average pooling
    gradcam = torch.relu(torch.sum(weights * activations, dim=1)).squeeze().numpy()  # Weighted sum
    # Normalize heatmap
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

    # Resize heatmap to match original image
    #obs = obs["camera_rgb"]
    obs = obs.squeeze(0)
    obs = np.transpose(obs, (2,1,0))
    heatmap = cv2.resize(gradcam, (obs.shape[0], obs.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    forward_handle.remove()
    backward_handle.remove()
    return heatmap