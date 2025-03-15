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
import pyrealsense2 as rs
import os
import requests

# Set QT environment variable to avoid issues on headless systems
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Fixed IP address of the robot
ROBOT_IP = "192.168.137.56"

# Get the target object from user input
TARGET_OBJECT = input("What are you looking for? \n")


def move_robot(command):
    url = f"http://{ROBOT_IP}/js?json={command}"
    response = requests.get(url)
    content = response.text

def get_object_direction(detections, depth, width, height, profile):
    min_distance = float("inf")
    direction = "Stop"
    object_position = None
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.9:
            label = int(detections[0, 0, i, 1])
            className = classNames[label]

            if className == TARGET_OBJECT:
                xmin = int(detections[0, 0, i, 3] * width)
                ymin = int(detections[0, 0, i, 4] * height)
                xmax = int(detections[0, 0, i, 5] * width)
                ymax = int(detections[0, 0, i, 6] * height)
                center_x = (xmin + xmax) / 2

                object_depth = depth[ymin:ymax, xmin:xmax].astype(float) * depth_scale
                dist, _, _, _ = cv2.mean(object_depth)

                if dist < min_distance:
                    min_distance = dist
                    object_position = center_x

    if object_position is not None:
        third_width = width / 3
        if object_position < third_width:
            direction = "Turn left"
        elif object_position > 2 * third_width:
            direction = "Turn right"
        else:
            direction = "Go straight ahead"

        print(f"Nearest {TARGET_OBJECT} detected at {min_distance:.3f} meters. {direction}")

        if min_distance <= 0.2:
            direction = "Stop"

        if direction == "Turn left":
            move_robot('{"T":1,"L":-0.1,"R":0.1}')
        elif direction == "Turn right":
            move_robot('{"T":1,"L":0.1,"R":-0.1}')
        elif direction == "Go straight ahead":
            move_robot('{"T":1,"L":0.2,"R":0.2}')
        elif direction == "Stop":
            move_robot('{"T":0,"L":0.0,"R":0.0}')

    return direction


pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)
colorizer = rs.colorizer()

prototxt_path = "/home/robi/fourth_semester_project/MobileNetSSD_deploy.prototxt"
model_path = "/home/robi/fourth_semester_project/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse",
              "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor")

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

        blob = cv2.dnn.blobFromImage(color, 0.007843, (300, 300), 127.53, False)
        net.setInput(blob, "data")
        detections = net.forward("detection_out")

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.9:
                label = int(detections[0, 0, i, 1])
                className = classNames[label]

                if className in allowed_objects:
                    direction = get_object_direction(detections, depth, width, height, profile)
                    if direction == "Stop":
                        print("Target reached. Stopping...")
                        pipe.stop()
                        cv2.destroyAllWindows()
                        exit()

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception as e:
        print(f"Error: {e}")
        continue

pipe.stop()
cv2.destroyAllWindows()
