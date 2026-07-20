# Optional ROS 2 rosbag bridge

이 폴더는 SO-101 follower 관절값(`sensor_msgs/JointState`)과 RGB image topic을 rosbag2로
기록하고, 기록된 관절값을 시간 순서대로 follower에 다시 보내는 기존 ROS 2 Jazzy 패키지다.
학습용 데이터는 최상위 `launchfiles/collect_data.bash`를 사용한다.

```bash
cd /path/to/so101-ros2
source /opt/ros/jazzy/setup.bash
source scripts/setup_env.bash
./libraries/ros2_bridge/scripts/build.bash
source install/setup.bash
export FOLLOWER_PORT=/dev/serial/by-id/CHANGE_ME
```

Terminal 1에서 torque-off 상태/카메라를 발행한다.

```bash
ros2 launch so101_ros2 record.launch.py port:="$FOLLOWER_PORT"
```

Terminal 2에서 관절만 또는 관절+카메라를 기록한다.

```bash
./libraries/ros2_bridge/scripts/record_bag.bash pick_01
./libraries/ros2_bridge/scripts/record_bag.bash pick_01_camera \
  /so101/record/camera/wrist/image_raw
```

재생은 먼저 `dry_run`으로 검증한다. Terminal 1에서 subscriber, Terminal 2에서 bag을 실행한다.

```bash
ros2 launch so101_ros2 replay.launch.py dry_run:=true
./libraries/ros2_bridge/scripts/play_bag.bash data/rosbags/pick_01
```

실제 arm 재생은 중립 자세·비상 정지를 확인한 뒤 `dry_run:=false port:="$FOLLOWER_PORT"`로
실행한다. 시작 자세 차이와 frame 간 급격한 변화는 latch fault로 거부된다.
