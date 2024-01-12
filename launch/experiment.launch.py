from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    # Experiment Type Args
    conversation = True  # Set to True or False to launch experiment_conversation or experiment_simple
    gripper_enabled = True  # Set to True or False based on your requirement

    return LaunchDescription([
        # Launch Conversation Experiment Node
        Node(
            package='multimodal_fusion',
            executable='experiment_conversation.py',
            name='experiment',
            output='screen',
            parameters=[{'gripper_enabled': gripper_enabled}],
            condition=IfCondition(value=conversation)
        ),

        # Launch Experiment Node
        Node(
            package='multimodal_fusion',
            executable='experiment_simple.py',
            name='experiment',
            output='screen',
            parameters=[{'gripper_enabled': gripper_enabled}],
            condition=UnlessCondition(value=conversation)
        )
    ])