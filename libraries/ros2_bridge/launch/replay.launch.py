"""Launch the guarded SO-101 rosbag replay subscriber."""

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
            description="Follower serial port; not required while dry_run is true",
        ),
        DeclareLaunchArgument("robot_id", default_value="osy_so101_follower"),
        DeclareLaunchArgument("calibration_dir", default_value=""),
        DeclareLaunchArgument(
            "dry_run",
            default_value="true",
            description="Validate bag commands without opening or moving the arm",
        ),
        DeclareLaunchArgument("max_relative_target_deg", default_value="5.0"),
        DeclareLaunchArgument("max_start_delta_rad", default_value="0.3490658504"),
        DeclareLaunchArgument("max_start_delta_gripper", default_value="0.25"),
        DeclareLaunchArgument("max_step_delta_rad", default_value="0.2617993878"),
        DeclareLaunchArgument("max_step_delta_gripper", default_value="0.25"),
    ]
    node = Node(
        package="so101_ros2",
        executable="so101_hardware",
        name="so101_hardware",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "mode": "replay",
                "dry_run": ParameterValue(LaunchConfiguration("dry_run"), value_type=bool),
                "port": LaunchConfiguration("port"),
                "robot_id": LaunchConfiguration("robot_id"),
                "calibration_dir": LaunchConfiguration("calibration_dir"),
                "command_topic": "/so101/record/joint_states",
                "max_relative_target_deg": ParameterValue(
                    LaunchConfiguration("max_relative_target_deg"), value_type=float
                ),
                "max_start_delta_rad": ParameterValue(
                    LaunchConfiguration("max_start_delta_rad"), value_type=float
                ),
                "max_start_delta_gripper": ParameterValue(
                    LaunchConfiguration("max_start_delta_gripper"), value_type=float
                ),
                "max_step_delta_rad": ParameterValue(
                    LaunchConfiguration("max_step_delta_rad"), value_type=float
                ),
                "max_step_delta_gripper": ParameterValue(
                    LaunchConfiguration("max_step_delta_gripper"), value_type=float
                ),
            }
        ],
    )
    return LaunchDescription([*arguments, node])
