# 🤖 Robot Mapping and Object Tracking

## 🗺️ Main Project: Room Mapping Robot
The goal of this project is to develop a robot capable of mapping an entire room. The generated map will then be uploaded to a database, allowing other users to download and utilize it for their own robots.

### Hardware in Use:
- **Waveshare UGV02** 
- **Raspberry Pi 5** + Active Cooler
  - Ubuntu Noble (Pro) 24.04
- **A2M8 LiDAR Sensor**

You can explore the code for this project [here](https://github.com/Spodymun/ros2-lidar-explorer).

---

## 🎯 Side Project: Object Tracking and Pursuit
Additionally, the robot should be capable of tracking and pursuing a specific object within the mapped area. This feature builds on the mapping system to enhance interaction with objects in the environment.

### Hardware in Use:
- **Waveshare UGV02**
- **Raspberry Pi 5** + Active Cooler
  - Ubuntu Noble (Pro) 24.04
- **Intel RealSense D415 Camera**

### Base Code (Verified as of 01.02.2025):
- **Intel RealSense**: [GitHub Repo](https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb)
  
### Object Detection Model: MobileNetSSD (Verified as of 01.02.2025)
- **Model Files**:
  - `deploy.prototxt`: MobileNetSSD architecture
  - `mobilenet_iter_73000.caffemodel`: Pre-trained weights

You can download the model files from [MobileNet-SSD GitHub](https://github.com/chuanqi305/MobileNet-SSD).

---
