# ==============================================
# 📌 Project Information
# ==============================================
# 🔹 Base Code: Intel RealSense (GitHub)
#    → https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
#    (Verified as of 01.02.2025)
#
# 🔹 Object Detection Model: MobileNetSSD
#    → Downloaded from: https://github.com/chuanqi305/MobileNet-SSD
#    - Model Files:
#      - deploy.prototxt → MobileNetSSD architecture
#      - mobilenet_iter_73000.caffemodel → Pre-trained weights
#
# 🔹 PyRealSense2 for Raspberry Pi 5
#    → Currently, there is no "official" method from Intel, but this approach worked for me:
#    - https://www.robotexchange.io/t/how-to-setup-the-intel-realsense-software-and-pyrealsense2-library-in-ubuntu-on-a-raspberryi-pi-5/3414
# ==============================================

import cv2
import numpy as np
import pyrealsense2 as rs #Python Version 3.11 (or older) needed
import os
import requests

# Set QT environment variable to avoid issues on headless systems
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Fixed IP address of the robot
ROBOT_IP = "192.168.178.53"

# Get the target object from user input
TARGET_OBJECT = input("What are you looking for? \n")

#Sends control commands to the robot via HTTP request
def move_robot(command):
    url = f"http://{ROBOT_IP}/js?json={command}"
    response = requests.get(url)
    content = response.text

#Determines the direction towards the target object and sends commands to the robot
def get_bottle_direction(detections, depth, crop_start, expected, width, profile):
    min_distance = float("inf")
    direction = "Stop"
    bottle_position = None
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.9:  # Confidence threshold
            label = int(detections[0, 0, i, 1])
            className = classNames[label]

            if className == TARGET_OBJECT:
                xmin = int(detections[0, 0, i, 3] * expected)
                xmax = int(detections[0, 0, i, 5] * expected)
                center_x = (xmin + xmax) / 2

                # Map detected object region to depth image
                scale = height / expected
                xmin_depth = int((xmin + crop_start) * scale)
                xmax_depth = int((xmax + crop_start) * scale)
                object_depth = depth[:, xmin_depth:xmax_depth].astype(float) * depth_scale
                dist, _, _, _ = cv2.mean(object_depth)

                if dist < min_distance:  # Keep the nearest object
                    min_distance = dist
                    bottle_position = center_x

    if bottle_position is not None:
        third_width = expected / 3  # Divide image into left, center, right
        if bottle_position < third_width:
            direction = "Turn left"
        elif bottle_position > 2 * third_width:
            direction = "Turn right"
        else:
            direction = "Go straight ahead"

        print(f"Nearest {TARGET_OBJECT} detected at {min_distance:.3f} meters. {direction}")

        # Stop if close enough
        if min_distance <= 0.2:
            direction = "Stop"

        # Send command to robot based on detected direction
        if direction == "Turn left":
            move_robot('{"T":1,"L":-0.1,"R":0.1}')  # Slow left turn
        elif direction == "Turn right":
            move_robot('{"T":1,"L":0.1,"R":-0.1}')  # Slow right turn
        elif direction == "Go straight ahead":
            move_robot('{"T":1,"L":0.2,"R":0.2}')  # Slow forward movement
        elif direction == "Stop":
            move_robot('{"T":0,"L":0.0,"R":0.0}')  # Stop

    return direction

# Setup RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()

# Define the target image size (expected height in pixels)
expected = 300  # Set image height (e.g., 300px)

# Enable depth and color streams
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)
colorizer = rs.colorizer()

# Load MobileNet SSD model
prototxt_path = "/home/robi/fourth_semester_project/MobileNetSSD_deploy.prototxt"
model_path = "/home/robi/fourth_semester_project/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Class names from MobileNet SSD
classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse",
              "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor")

# Allowed objects for detection
allowed_objects = {"chair", "person", "bottle"}

while True:
    try:
        frameset = pipe.wait_for_frames()
        frameset = align.process(frameset)

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        height, width = color.shape[:2]
        aspect = width / height
        resized_image = cv2.resize(color, (round(expected * aspect), expected))
        crop_start = round(expected * (aspect - 1) / 2)
        crop_img = resized_image[0:expected, crop_start:crop_start+expected]

        blob = cv2.dnn.blobFromImage(crop_img, 0.007843, (expected, expected), 127.53, False)
        net.setInput(blob, "data")

        # Get model detections
        detections = net.forward("detection_out")

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.9:
                label = int(detections[0, 0, i, 1])
                className = classNames[label]

                if className in allowed_objects:
                    direction = get_bottle_direction(detections, depth, crop_start, expected, width, profile)

                    if direction == "Stop":
                        print("Target reached. Stopping...")
                        pipe.stop()
                        cv2.destroyAllWindows()
                        exit()

        # Exit loop if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception as e:
        print(f"Error: {e}")
        continue

# Stop RealSense pipeline and close windows
pipe.stop()
cv2.destroyAllWindows()