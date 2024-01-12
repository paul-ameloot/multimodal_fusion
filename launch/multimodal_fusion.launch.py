from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch Fusion Node
        Node(
            package='multimodal_fusion',
            executable='fusion.py',
            name='multimodal_fusion',
            output='screen'
        ),
    ])