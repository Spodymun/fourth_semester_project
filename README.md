📌 Project Information
==============================================

Main Project:
- The goal of this project is to develop a robot capable of mapping an entire room. The generated map will then be uploaded to a database, allowing other users to download and utilize it for their own robots.

🔹 Hardware in Use:
   - Waveshare UGV02
   - Raspberry Pi 5 + Active Cooler
   - (Intel RealSense D415 camera)
   - A2M8 - R4 LiDAR sensor

🔹 Ros 2 Installation (Verified as of 03.03.2025)
   - https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

🔹 Creating a Workspace on your pi (Verified as of 14.03.2025)
   - https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html
   - Some Tips:   
     - My Workspace is called "ws_lidar", you probably wanna keep it that way so that there is no problem when using my scripts
     - When you've done that you can clone my repo in your "~/ws_lidar/src" folder 
       -> https://github.com/Spodymun/ros2-lidar-explorer
   
==============================================

Side Project:
- Additionally, the robot should be able to track and pursue a specific object within the mapped area.

🔹 Hardware in Use:
   - Waveshare UGV02
   - Raspberry Pi 5 + Active Cooler
   - Intel RealSense D415 camera

🔹 Base Code: Intel RealSense (GitHub) (Verified as of 01.02.2025)
   - https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb

🔹 Object Detection Model: MobileNetSSD (Verified as of 01.02.2025)
   - Downloaded from: https://github.com/chuanqi305/MobileNet-SSD
   - Model Files:
     - deploy.prototxt → MobileNetSSD architecture
     - mobilenet_iter_73000.caffemodel → Pre-trained weights

🔹 PyRealSense2 for Raspberry Pi 5
   - Currently, there is no "official" method from Intel, but this approach worked for me (Verified as of 01.02.2025) :
   - https://www.robotexchange.io/t/how-to-setup-the-intel-realsense-software-and-pyrealsense2-library-in-ubuntu-on-a-raspberryi-pi-5/3414

==============================================
