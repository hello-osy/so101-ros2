# SO-101: Orin 데이터 수집 → RTX 5070 Ti 학습 → Orin 실행

SO-101 leader를 움직여 follower를 조종하고, 카메라가 포함된 demonstration을 수집한 뒤
SmolVLA를 LoRA 학습하여 follower에서 실행하는 Jetson Orin Nano용 프로젝트다.

사용자가 수정할 설정 파일은 [`config/system.yaml`](config/system.yaml) 하나뿐이다. 장치,
카메라, 데이터셋, 모델, 학습값을 이 파일에서 통합 관리하며 모든 주요 명령은 이 파일 하나를
인자로 받는다.

## 한눈에 보는 구조

```text
config/system.yaml             # 유일한 사용자 설정
launchfiles/                   # 한 줄 실행 명령
scripts/system_config.py       # 통합 설정을 작업별 LeRobot 설정으로 변환
scripts/calibration/           # 캘리브레이션
scripts/data_collection/       # demonstration 수집
scripts/training/              # LoRA 학습
scripts/inference/             # 실시간 추론과 성능 측정
data/                          # dataset, model, calibration, log
libraries/lerobot/             # bootstrap이 설치하는 고정 LeRobot 소스
```

실행할 때마다 원본 `system.yaml`, 변환된 LeRobot YAML, 실제 명령, console log, Git/CUDA/하드웨어
정보가 해당 run의 `artifacts/`에 저장된다. Orin과 데스크탑이 같은 저장소를 각각 clone하고,
Git에 넣지 않는 대용량 데이터만 SSH/rsync로 주고받는다.

## 1. 최초 설정

프로젝트를 받는다.

```bash
git clone https://github.com/hello-osy/so101-ros2.git && cd so101-ros2
```

Ubuntu 패키지를 설치한다.

```bash
./scripts/install_system_dependencies.bash
```

JetPack에 맞는 NVIDIA PyTorch와 torchvision wheel을 설치한 뒤 버전을 확인한다. 기본 기대
버전은 `config/system.yaml`의 `system.torch`, `system.torchvision`, `system.cuda`에 있다.

```bash
python3 -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda)'
```

LeRobot과 Python 환경을 설치한다.

```bash
./scripts/bootstrap_environment.bash config/system.yaml
```

SmolVLA base model과 VLM metadata를 받는다.

```bash
./launchfiles/download_models.bash config/system.yaml
```

설치와 설정을 하드웨어 없이 검사한다.

```bash
./scripts/test_project.bash config/system.yaml
```

프로젝트 폴더를 옮겨서 가상환경 경로가 깨졌을 때만 복구한다.

```bash
./scripts/repair_environment.bash
```

## 2. `config/system.yaml`에서 확인할 값

최초 실행 전에 다음 항목을 실제 환경에 맞춘다.

| YAML 위치 | 의미 |
|---|---|
| `runs.camera_viewer` | 동시에 표시할 카메라 이름 목록 |
| `devices.follower` | 움직일 follower의 포트, ID, calibration 저장 위치, 안전 제한 |
| `devices.leader` | 손으로 조작할 leader의 포트, ID, calibration 저장 위치 |
| `devices.cameras` | 카메라 장치, 해상도, FPS, 영상 포맷 |
| `dataset` | 만들 dataset의 ID, 작업 문장, episode 수와 저장 옵션 |
| `model` | 다운로드할 base/VLM, 학습할 모델, LoRA, 실행할 checkpoint |
| `runs.calibration` | calibration 대상과 통신 재시도 횟수 |
| `runs.teleoperation` | 직접 조종 주기와 저지연 카메라 미리보기 |
| `runs.training` | step, batch, worker, checkpoint 주기 |
| `runs.training_desktop` | RTX 5070 Ti 학습 batch, LoRA, 검증, checkpoint 주기 |
| `transfer.desktop` | 데스크탑 SSH 주소와 데스크탑의 repo 절대 경로 |
| `runs.inference` | 실행 시간, 보간, compile, latency 측정 |
| `runs.benchmark`, `runs.profiling` | 반복 횟수와 Torch/Nsight 측정 옵션 |

현재 기본 장치값은 follower `/dev/ttyACM1`, leader `/dev/ttyACM0`, wrist camera
`/dev/video0`의 MJPG 640×480 30 FPS, front camera `/dev/video2`의 YUYV 640×480 30 FPS다.
재부팅에도 장치명이 유지되게 하려면 `/dev/serial/by-id/...` 경로를 권장한다.

장치와 카메라를 찾는 명령은 다음과 같다.

```bash
./libraries/venv/bin/lerobot-find-port
```

```bash
./libraries/venv/bin/lerobot-find-cameras opencv
```

카메라가 지원하는 픽셀 포맷, 해상도, FPS 조합을 확인한다.

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

현재 적용된 포맷과 노출·초점 등의 전체 설정을 확인한다.

```bash
v4l2-ctl --device=/dev/video0 --all
```

변경 가능한 노출·초점·밝기 등의 control 범위를 확인한다.

```bash
v4l2-ctl --device=/dev/video0 --list-ctrls-menus
```

USB 권한이 없으면 사용자를 `dialout` 그룹에 추가한 후 로그아웃하고 다시 로그인한다.

```bash
sudo usermod -aG dialout "$USER"
```

## 3. 실행 순서

팔 전원을 켜기 전에 leader와 follower를 비슷한 중립 자세에 둔다. 케이블 걸림, 이상한 저항,
급격한 동작이 있으면 즉시 `Ctrl+C`를 누르거나 모터 전원을 차단한다.

### 카메라만 보기

```bash
./launchfiles/view_camera.bash config/system.yaml
```

leader와 follower 포트를 열지 않고 `runs.camera_viewer.cameras`에 지정한 `wrist`, `front`
카메라를 각각 저지연 창에 표시한다. 터미널에서 `Ctrl+C`를 누르면 두 창이 모두 종료된다.

### 3.1 캘리브레이션

```bash
./launchfiles/calibrate.bash config/system.yaml
```

`runs.calibration.target`의 기본값은 `both`이므로 follower와 leader를 차례로 캘리브레이션한다.
기존 값을 다시 만들려면 질문에서 `c`와 Enter를 누르고, 안내에 따라 각 관절과 gripper를
천천히 전체 범위로 움직인다. 결과는 `data/calibration`에, 실행 로그는
`data/calibration_runs`에 저장된다.

### 3.2 Leader로 follower 조종 + 실시간 카메라

```bash
./launchfiles/teleoperate.bash config/system.yaml
```

leader를 손으로 움직이면 follower가 60Hz 제어 주기로 따라간다. wrist 카메라는 별도의
`ffplay` 창에서 30Hz로 표시하므로 영상 처리가 모터 제어를 막지 않는다. 영상이 필요 없으면
`runs.teleoperation.camera_preview: null`로 바꾼다. Rerun 관절 표시가 꼭 필요할 때만
`display_data: true`로 바꾼다. 조종 종료는 명령을 실행한 터미널에서 `Ctrl+C`를 누른다.

`devices.follower.max_relative_target: 5.0`은 한 control tick의 최대 관절 변화량을 5도로
제한한다. 이 값은 직접 조종, 데이터 수집, 추론에 공통 적용된다.

### 3.3 Demonstration 데이터 수집

```bash
./launchfiles/collect_data.bash config/system.yaml
```

키보드는 다음처럼 사용한다.

- `Enter`: 첫 episode 시작
- `→` 1회: 현재 episode 녹화 종료 후 reset 단계로 이동
- 물체와 로봇을 시작 상태로 reset한 뒤 `→` 1회: 현재 episode 저장 후 다음 녹화 시작
- `←`: 현재 episode 폐기 후 다시 수집
- `Esc`: 전체 수집 종료 및 dataset 저장

수집 중 Rerun 창에는 녹화 프로세스가 이미 읽은 wrist/front 영상이 실시간 표시된다. 카메라
장치를 별도 프로그램이 중복해서 열지 않으므로 녹화 영상과 시각화 영상이 동일하다. 터미널에는
아래 상태가 episode 단계마다 반복해서 표시된다.

```text
[수집 상태] 완료 3/50 | 현재 episode 4 녹화 중
[키] →: 녹화 종료 | ←: 현재 episode 폐기/재촬영 | Esc: 전체 종료
[다음 단계] 녹화를 끝내려면 →, reset을 마쳤으면 다시 →
```

`runs.collection.show_clamp_warnings: false`는 반복되는 관절 clamp 로그만 숨긴다. follower의
`max_relative_target` 안전 제한과 실제 전송 action 저장은 그대로 적용된다. 경고를 다시
확인하려면 이 값을 `true`로 바꾼다.

결과는 `data/collected_datasets/<run>/dataset`에 저장되고, 가장 최근 성공 결과는
`data/collected_datasets/latest/dataset`으로 연결된다. 데이터 형식은 LeRobotDataset v3이며
관절값은 Parquet, 카메라는 MP4, task와 통계는 metadata로 저장된다.

데스크탑으로 30초마다 증분 전송하면서 수집하려면 아래 한 줄을 대신 실행한다.

```bash
./launchfiles/collect_and_sync.bash config/system.yaml
```

이미 수집한 데이터만 다시 동기화할 수도 있다.

```bash
./launchfiles/sync_dataset.bash config/system.yaml
```

### 3.4 SmolVLA LoRA 학습

```bash
./launchfiles/train.bash config/system.yaml
```

기본값은 Orin 8 GB를 위한 `batch_size: 1`, `num_workers: 0`, AMP, LoRA다. 처음에는
`runs.training.steps: 20`으로 smoke test한 뒤 본 학습 횟수로 늘리는 편이 안전하다. 결과는
`data/training_outputs/<run>/training`에 저장되고 최신 성공 결과는
`data/training_outputs/latest`로 연결된다.

### 3.5 학습 모델 실시간 실행

```bash
./launchfiles/inference.bash config/system.yaml
```

사용할 checkpoint는 `model.trained_policy_path`에서 지정한다. 각 control decision latency는
`latency.jsonl`, 요약 통계는 `latency_summary.json`에 저장된다. 추론 중 카메라를 표시하려면
`runs.inference.display_data: true`로 바꾼다.

## 4. RTX 5070 Ti 데스크탑에서 파인튜닝

데스크탑은 Ubuntu 24.04, Python 3.12, NVIDIA driver가 설치되어 있고 `nvidia-smi`가 정상이어야
한다. PyTorch wheel에 CUDA runtime이 포함되므로 별도의 CUDA Toolkit은 학습에 필수는 아니다.
저장소를 pull한 뒤 필요한 OS 패키지를 설치한다.

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3.12-dev build-essential git git-lfs ffmpeg rsync openssh-server
```

Orin이 파일을 보내고 가져갈 수 있도록 데스크탑 SSH server를 켠다.

```bash
sudo systemctl enable --now ssh
```

5070 Ti 전용 가상환경, CUDA 13.2 PyTorch wheel, 고정 LeRobot revision을 설치한다.

```bash
./scripts/bootstrap_desktop.bash config/system.yaml
```

base model과 VLM metadata를 준비한다.

```bash
./launchfiles/download_models_desktop.bash config/system.yaml
```

Orin에서 전송된 dataset이 `data/collected_datasets/latest/dataset`에 있는지 확인한 뒤 학습한다.

```bash
./launchfiles/train_desktop.bash config/system.yaml
```

기본 recipe는 16GB VRAM을 고려한 AMP, `batch_size: 8`, vision encoder 고정, LoRA r64,
20,000 step, 10% episode validation이다. 먼저 `steps: 20`으로 smoke test한다. 이후
`nvidia-smi`에서 학습 중 여유 VRAM이 2GB 이상이면 `runs.training_desktop.batch_size`만 12,
그다음 16으로 올린다. OOM이면 4로 낮춘다. `compile_model`은 첫 실행 안정성을 위해 꺼져 있다.

학습이 끝난 뒤 **Orin에서** 다음 한 줄을 실행하면 마지막 checkpoint를 timestamp 폴더로 받고
`data/models/smolvla_finetuned/latest`를 원자적으로 갱신한다. 이후 기존 inference 명령이 바로
이 모델을 사용한다.

```bash
./launchfiles/pull_trained_model.bash config/system.yaml
```

## 5. Orin ↔ 데스크탑 SSH 설정

현재 Orin의 `enP8p1s0`은 UP이지만 IPv4가 없고 link-local IPv6만 있으므로, 지금 표시된
Wi-Fi neighbor만으로 어느 장치가 데스크탑인지 확정할 수 없다. 데스크탑에서 주소를 확인한다.

```bash
ip -br address && hostname -I
```

같은 LAN의 유선 IPv4를 `config/system.yaml`의 `transfer.desktop.host`에, 데스크탑 계정과 repo
절대 경로를 `user`, `repo_path`에 적는다. `ssh-copy-id: No identities found`가 나오면 Orin에
SSH key가 아직 없는 것이므로 먼저 생성한다. 질문에는 Enter를 눌러 기본 저장 위치를 쓰고,
자동 dataset 전송이 필요하면 passphrase도 비워 둔다.

```bash
ssh-keygen -t ed25519 -a 100 -C "so101-orin-to-desktop"
```

공개키를 데스크탑에 한 번 등록한다. 아래 `osy`는 `transfer.desktop.user`와 같은 실제
데스크탑 사용자명이어야 한다.

```bash
ssh-copy-id -p 22 osy@<DESKTOP_IP>
```

```bash
ssh -p 22 osy@<DESKTOP_IP> 'hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

### 재부팅 후에도 유지되는 직결 랜선 설정

학교 내부망의 공인 IP 대신 Orin과 데스크탑을 랜선으로 직접 연결한다. 두 장치에 다음 주소를
고정하면 Wi-Fi는 인터넷용으로 유지되고 dataset/model/profiling 전송만 유선으로 지나간다.

```text
데스크탑 Ethernet  192.168.50.1/24
Orin enP8p1s0      192.168.50.2/24
Gateway/DNS        없음
```

Orin에서 장치와 연결 이름을 확인한다.

```bash
nmcli device status
```

현재 Orin의 유선 연결 이름은 `Wired connection 1`이다. DHCP를 기다리는 이 프로필을 고정 IP
프로필로 바꾸고 부팅 시 자동 연결되도록 저장한다.

```bash
sudo nmcli connection modify "Wired connection 1" ipv4.method manual ipv4.addresses 192.168.50.2/24 ipv4.gateway "" ipv4.dns "" ipv4.never-default yes ipv6.method disabled connection.autoconnect yes connection.autoconnect-retries 0
```

```bash
sudo nmcli connection up "Wired connection 1"
```

데스크탑에서도 유선 장치와 연결 이름을 확인한다. `enp...` 또는 `eno...`가 보통 유선 장치다.

```bash
nmcli device status
```

출력의 실제 유선 연결 이름을 `<DESKTOP_WIRED_CONNECTION>` 대신 넣는다.

```bash
sudo nmcli connection modify "<DESKTOP_WIRED_CONNECTION>" ipv4.method manual ipv4.addresses 192.168.50.1/24 ipv4.gateway "" ipv4.dns "" ipv4.never-default yes ipv6.method disabled connection.autoconnect yes connection.autoconnect-retries 0
```

```bash
sudo nmcli connection up "<DESKTOP_WIRED_CONNECTION>"
```

데스크탑 유선 장치의 `CONNECTION` 값이 `--`라서 수정할 프로필이 없다면 `<DESKTOP_ETH>`에
장치 이름을 넣어 새 프로필을 만든다.

```bash
sudo nmcli connection add type ethernet ifname <DESKTOP_ETH> con-name so101-direct ipv4.method manual ipv4.addresses 192.168.50.1/24 ipv4.never-default yes ipv6.method disabled connection.autoconnect yes connection.autoconnect-retries 0
```

```bash
sudo nmcli connection up so101-direct
```

데스크탑에서 SSH server도 부팅할 때 자동 시작되게 한다.

```bash
sudo systemctl enable --now ssh
```

데스크탑 방화벽이 활성화되어 있다면 직결 대역의 SSH만 허용한다.

```bash
sudo ufw allow from 192.168.50.0/24 to any port 22 proto tcp
```

Orin에서 주소와 유선 route를 확인한다.

```bash
ip -br -4 address show enP8p1s0
```

```bash
ip route get 192.168.50.1
```

route 출력에 `dev enP8p1s0 src 192.168.50.2`가 나오면 정상이다. 통신과 GPU를 최종
확인한다.

```bash
ping -c 3 192.168.50.1
```

```bash
ssh -o ConnectTimeout=5 osy@192.168.50.1 'hostname && nvidia-smi'
```

`nmcli connection modify`로 저장한 프로필과 `systemctl enable`로 등록한 SSH service는 재부팅
후에도 유지된다. 양쪽을 재부팅한 뒤 위 `ping`과 `ssh` 명령이 다시 성공하면 설정이 완료된
것이다. 프로젝트의 `transfer.desktop.host` 기본값도 `192.168.50.1`로 설정되어 있다.

## 6. 성능 측정

모든 측정은 실제 wrist/front 카메라 관측과 checkpoint를 사용하는 라이브 rollout에서 수행한다.
따라서 **follower 로봇이 실제로 움직이며**, 작업 공간과 비상 정지 수단을 준비해야 한다. 별도
warmup sample은 제외하지 않고 첫 실제 action부터 latency·메모리·tegrastats를 측정한다.
일반/Torch 측정은 Ctrl-C까지 계속된다. Nsys/NCU는 실제 rollout 동안 profiler를 대기시켜 GPU를
warm 상태로 만든 뒤, Ctrl-C 한 번으로 action 전송 차단 → RTC queue 제거 → 동일 프로세스의 최신
관측/RTC prefix 상세 capture → 종료/전송 순서로 동작한다.
상세 capture가 끝나기를 기다릴 수 없으면 Ctrl-C를 한 번 더 누른다. 이때 `sudo`, Nsys/NCU,
rollout Python이 속한 전체 process group을 종료하며, 5초 뒤에도 남아 있으면 자동으로 강제
종료한다. 그 전에 Ctrl-C를 추가로 누르면 즉시 강제 종료한다.
아직 첫 실제 inference가 완료되기 전에 Ctrl-C를 누르면 보존할 warm 상태가 없으므로 상세
capture를 시작하지 않고 전체 profiler process group을 종료한다.

최초 한 번 NVIDIA GPU performance counter를 일반 사용자에게 허용하고 재부팅한다. 권한이
막힌 상태에서는 profiler가 로봇을 연결하기 전에 설정 방법을 출력하고 종료한다.

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
  sudo tee /etc/modprobe.d/so101-profiler.conf
sudo update-initramfs -u
sudo reboot
```

재부팅 후 값이 `0`인지 확인한다.

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params
```

Jetson/Tegra의 CUDA trace는 이 값이 `0`이어도 현재 CUPTI에서 root 권한을 요구할 수 있다.
성능 측정 launcher는 profiler가 포함된 rollout 자식만 `sudo`로 실행하므로 시작할 때 암호를 한 번
입력한다. report 전송은 계속 원래 사용자 계정의 SSH 설정을 사용한다. launcher 전체를
`sudo ./launchfiles/...`로 실행하지 않는다.

일반 latency와 메모리를 측정한다.

```bash
./launchfiles/benchmark_inference.bash config/system.yaml
```

PyTorch operator와 Chrome trace를 기록한다.

```bash
./launchfiles/profile_torch.bash config/system.yaml
```

Jetson nightly PyTorch/Kineto에서 RTC 백그라운드 CUDA activity를 직접 수집하면 CUDA context가
불안정해질 수 있어 기본 Torch trace는 CPU operator dispatch를 기록한다. CUDA timeline은 아래
Nsight Systems 명령으로, kernel metric은 Nsight Compute 명령으로 기록한다. 실험적으로 Torch
CUDA activity를 다시 켜려면 `runs.profiling.torch.cuda_activity: true`로 바꿀 수 있다.

Nsight Systems report를 기록한다.

```bash
./launchfiles/profile_nsys.bash config/system.yaml
```

실행 직후에는 CUDA capture가 꺼져 있어 실제 로봇 rollout의 timing을 교란하지 않는다. 충분히
동작시킨 뒤 Ctrl-C를 누르면 새 action 전송을 즉시 차단하고 RTC queue를 비운다. 모델, CUDA
context, allocator/cache, 최신 실제 observation과 RTC leftover는 유지한 채 기본 1회만 CUDA
activity를 capture한다. capture 결과 action은 로봇에 보내지 않고 폐기한다.

Nsight Compute kernel report를 기록한다.

```bash
./launchfiles/profile_ncu.bash config/system.yaml
```

NCU도 같은 safe transition을 사용한다. 기본값은 한 번의 warmed inference에서 앞쪽 kernel 10개만
수집한다. `runs.profiling.safe_gpu.iterations`와 `runs.profiling.ncu.launch_count`를 늘리면 report와
정지 시간이 크게 증가하므로 물리 로봇에서는 기본값을 권장한다.

가장 최근 profiling run 전체(`.ncu-rep`/`.nsys-rep`/Torch trace, 설정, console, 측정값)를 데스크탑 repo의
`data/profiling_from_orin/<ORIN_HOSTNAME>/`으로 보낸다.

```bash
./launchfiles/push_ncu_results.bash config/system.yaml
```

모든 profiler 형식에 맞는 이름의 동일 명령도 제공한다.

```bash
./launchfiles/push_profile_results.bash config/system.yaml
```

데스크탑에서 가장 최근 report를 자동 판별하여 NCU UI, Nsight Systems UI 또는 Perfetto를 실행한다.

```bash
./launchfiles/view_latest_profile.bash config/system.yaml
```

형식을 직접 지정할 수도 있다.

```bash
./launchfiles/view_latest_profile.bash config/system.yaml ncu
./launchfiles/view_latest_profile.bash config/system.yaml nsys
./launchfiles/view_latest_profile.bash config/system.yaml torch
```

AMP와 Nsight 옵션은 `config/system.yaml`의 `model.use_amp`, `runs.profiling`에서 바꾼다.
`tegrastats`가 있으면 unified RAM, clock, 온도, 전력도 Ctrl-C까지 동시에 저장된다.

## 7. Jetson 상태와 성능 모드

현재 CUDA, 메모리, 온도와 clock 상태를 확인한다.

```bash
./scripts/jetson_status.bash
```

보드가 지원하는 전원 모드 ID를 확인한다.

```bash
sudo nvpmodel -q --verbose
```

충분한 전원과 냉각을 준비한 뒤 선택한 모드와 최대 clock을 적용한다.

```bash
./scripts/jetson_performance.bash <MODE_ID>
```

## 기존 ROS 2 코드

기존 rosbag joint/image record/replay 패키지는 [`libraries/ros2_bridge`](libraries/ros2_bridge/README.md)에
분리해 보존했다. ROS 2 Jazzy와 colcon이 있는 환경에서만 함께 검사한다.

```bash
BUILD_ROS2=1 ./scripts/test_project.bash config/system.yaml
```
