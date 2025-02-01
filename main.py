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
#      - mobilenet_iter_73000.caffemodel → Pretrained weights
# ==============================================

import cv2
import numpy as np
import pyrealsense2 as rs #Python Version 3.11 (or older) needed

# Setup RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()

# Enable depth and color streams
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start pipeline
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)  # Align depth to color frame
colorizer = rs.colorizer()  # Colorize depth

# Load MobileNet SSD model
prototxt_path = "C:/Users/schra/PycharmProjects/fourth_semester_project/MobileNetSSD_deploy.prototxt"
model_path = "C:/Users/schra/PycharmProjects/fourth_semester_project/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Check if model loads correctly
try:
    net.getLayerNames()
    print("MobileNet SSD model loaded successfully!")
except:
    print("Error: Model failed to load!")

# Detection parameters
inScaleFactor = 0.007843
meanVal = 127.53
expected = 300  # Model input size

# Class names from MobileNet SSD
classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse",
              "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor")

# Define objects to detect
allowed_objects = {"chair", "person", "bottle"}  # Only detect these objects

while True:
    # Wait for frames and align them
    frameset = pipe.wait_for_frames()
    frameset = align.process(frameset)

    # Get color and depth frames
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    if not color_frame or not depth_frame:
        continue  # Skip if frames not available

    # Convert frames to numpy arrays
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    # Resize and crop for object detection
    height, width = color.shape[:2]
    aspect = width / height
    resized_image = cv2.resize(color, (round(expected * aspect), expected))
    crop_start = round(expected * (aspect - 1) / 2)
    crop_img = resized_image[0:expected, crop_start:crop_start+expected]

    # Prepare image for MobileNet SSD
    blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
    net.setInput(blob, "data")
    detections = net.forward("detection_out")

    # Process detected objects
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.9:  # Confidence threshold
            label = int(detections[0, 0, i, 1])
            className = classNames[label]

            # **Only process "chair", "person", or "bottle"**
            if className in allowed_objects:
                xmin = int(detections[0, 0, i, 3] * expected)
                ymin = int(detections[0, 0, i, 4] * expected)
                xmax = int(detections[0, 0, i, 5] * expected)
                ymax = int(detections[0, 0, i, 6] * expected)

                # Draw bounding box
                cv2.rectangle(crop_img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
                cv2.putText(crop_img, f"{className} {conf:.2f}", (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

                # Map detected object region to depth image
                scale = height / expected
                xmin_depth = int((xmin + crop_start) * scale)
                ymin_depth = int(ymin * scale)
                xmax_depth = int((xmax + crop_start) * scale)
                ymax_depth = int(ymax * scale)

                cv2.rectangle(depth, (xmin_depth, ymin_depth), (xmax_depth, ymax_depth), (255, 255, 255), 2)

                # Calculate depth for the detected object
                object_depth = depth[ymin_depth:ymax_depth, xmin_depth:xmax_depth].astype(float)
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                object_depth = object_depth * depth_scale
                dist, _, _, _ = cv2.mean(object_depth)

                print(f"Detected {className} at {dist:.3f} meters")

                # **Trigger message if bottle is 0.1 meters away**
                if className == "bottle" and 0.195 <= dist <= 0.205:
                    print("🔹 Hello, I reached a bottle!")
                    pipe.stop()  # Stop the RealSense pipeline
                    cv2.destroyAllWindows()  # Close all OpenCV windows
                    exit()  # Fully terminate the script

    # Show results
    cv2.imshow("Color Frame", crop_img)
    cv2.imshow("Depth Frame", cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
pipe.stop()
cv2.destroyAllWindows()