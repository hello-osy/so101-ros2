from setuptools import find_packages, setup

package_name = "so101_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [
            "launch/bridge.launch.py",
            "launch/record.launch.py",
            "launch/replay.launch.py",
        ]),
        (f"share/{package_name}/config", ["config/so101.example.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hello-osy",
    maintainer_email="hello-osy@users.noreply.github.com",
    description="Record and replay SO-101 follower joint states and camera images with ROS 2 bags.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["so101_hardware = so101_ros2.hardware_node:main"]},
)
