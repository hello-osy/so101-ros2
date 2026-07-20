"""Launch a bidirectional SO-101 bridge for other ROS controllers."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument(
            "port",
            default_value=EnvironmentVariable("FOLLOWER_PORT", default_value=""),
        ),
        DeclareLaunchArgument("robot_id", default_value="osy_so101_follower"),
        DeclareLaunchArgument("calibration_dir", default_value=""),
        DeclareLaunchArgument("publish_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("camera_configs_json", default_value="{}"),
        DeclareLaunchArgument("max_relative_target_deg", default_value="5.0"),
    ]
    node = Node(
        package="so101_ros2",
        executable="so101_hardware",
        name="so101_hardware",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "mode": "bridge",
                "dry_run": False,
                "port": LaunchConfiguration("port"),
                "robot_id": LaunchConfiguration("robot_id"),
                "calibration_dir": LaunchConfiguration("calibration_dir"),
                "publish_rate_hz": ParameterValue(
                    LaunchConfiguration("publish_rate_hz"), value_type=float
                ),
                "camera_configs_json": LaunchConfiguration("camera_configs_json"),
                "state_topic": "/so101/record/joint_states",
                "command_topic": "/so101/joint_commands",
                "camera_topic_prefix": "/so101/record/camera",
                "max_relative_target_deg": ParameterValue(
                    LaunchConfiguration("max_relative_target_deg"), value_type=float
                ),
            }
        ],
    )
    return LaunchDescription([*arguments, node])
