from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            parameters=[
                {'enable_color': True},
                {'enable_depth': True},
                {'align_depth.enable': True},
                {'pointcloud.enable': True},
                {'pointcloud.texture_stream': 'RS2_STREAM_COLOR'},
                {'color0.format': 'RGB8'},
                {'color0.profile': '1280x720x30'}
            ]
        )
    ])