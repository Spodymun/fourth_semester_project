==============================================
📌 Project Information
==============================================

🔹 Main Project:
The goal of this project is to develop a robot capable of mapping an entire room. The generated map will then be uploaded to a database, allowing other users to download and utilize it for their own robots.

🔹 Hardware in Use:
   → Waveshare UGV02
   → Raspberry Pi 5 + Active Cooler
   → (Intel RealSense D415 camera)
   → (2D/3D LiDAR sensor)

==============================================

🔹 Side Project:
Additionally, the robot should be able to track and pursue a specific object within the mapped area.

🔹 Hardware in Use:
   → Waveshare UGV02
   → Raspberry Pi 5 + Active Cooler
   → Intel RealSense D415 camera

🔹 Base Code: Intel RealSense (GitHub)
   → https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
   (Verified as of 01.02.2025)

🔹 Object Detection Model: MobileNetSSD
   → Downloaded from: https://github.com/chuanqi305/MobileNet-SSD
   - Model Files:
     - deploy.prototxt → MobileNetSSD architecture
     - mobilenet_iter_73000.caffemodel → Pre-trained weights

🔹 PyRealSense2 for Raspberry Pi 5
   → Currently, there is no "official" method from Intel, but this approach worked for me:
   - https://www.robotexchange.io/t/how-to-setup-the-intel-realsense-software-and-pyrealsense2-library-in-ubuntu-on-a-raspberryi-pi-5/3414

==============================================