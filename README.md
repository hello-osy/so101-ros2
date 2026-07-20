# SO-101 data → LoRA → inference

SO-101 leader/follower로 pick-and-place demonstration을 수집하고, SmolVLA를 LoRA
fine-tuning한 뒤 follower에서 실행하는 Jetson Orin Nano용 프로젝트다. 모든 실행은
`launchfiles/*.bash` 한 줄이고, 포트·카메라·학습값은 `scripts/configs/*.yaml`만 수정한다.

데이터 형식은 **LeRobotDataset v3.0**이다. 관절 state/action/timestamp는 Parquet,
카메라는 MP4, task·episode·정규화 통계는 JSON/Parquet metadata로 저장되어 LeRobot/Hugging
Face/PyTorch 학습기와 바로 호환된다.

## 구조

```text
launchfiles/                  # calibration, 수집, 학습, 추론, profiler 1-line entry
scripts/
  calibration/ data_collection/ training/ inference/
  models/ configs/ tests_for_codex/
libraries/
  lerobot/                    # bootstrap이 clone, Git 제외
  venv/                       # bootstrap이 생성, Git 제외
  patches/                    # 실행된 safe action을 기록하는 작은 LeRobot patch
  ros2_bridge/                # 기존 rosbag record/replay 코드
data/
  models/ training_outputs/ inference_logs/
  collected_datasets/ downloaded_datasets/
```

## 1. 환경 재현

기준 환경은 Ubuntu 24.04 arm64, Python 3.12, CUDA 13.2,
`torch==2.13.0+cu132`, `torchvision==0.28.0+cu132`이며 LeRobot은
`e40b58a8dfa9e7b86918c374791599d070518d11`로 고정된다.

```bash
sudo apt update
sudo apt install -y git git-lfs ffmpeg cmake build-essential python3-dev python3-venv \
  pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev

git clone https://github.com/hello-osy/so101-ros2.git
cd so101-ros2
```

JetPack에 맞는 NVIDIA PyTorch/torchvision wheel은 먼저 설치되어 있어야 한다. 일반 PyPI
CUDA wheel로 덮어쓰지 말고 다음 결과를 확인한다.

```bash
python3 -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda)'
```

그 다음 LeRobot clone, venv, Feetech/데이터/학습/SmolVLA/PEFT 의존성을 한 번에 만든다.
인터넷 연결이 필요하다.

```bash
./scripts/bootstrap_environment.bash
source scripts/setup_env.bash
```

프로젝트 폴더를 옮겼다면 절대경로가 남은 venv를 한 번 복구한다.

```bash
./scripts/repair_environment.bash
source scripts/setup_env.bash
```

USB 권한은 재로그인 후 적용된다.

```bash
sudo usermod -aG dialout "$USER"
ls -l /dev/serial/by-id/
./libraries/venv/bin/lerobot-find-port
./libraries/venv/bin/lerobot-find-cameras opencv
```

`/dev/ttyACM0` 대신 재부팅에도 유지되는 `/dev/serial/by-id/...`를 권장한다.

## 2. YAML에서 바꿀 것

최초 1회 다음 파일의 `CHANGE_ME_FOLLOWER`, `CHANGE_ME_LEADER`, `/dev/video0`을 실제 값으로
바꾼다. calibration과 모든 후속 실행에서 `id`와 `use_degrees`를 동일하게 유지한다.

- `scripts/configs/calibration.yaml`: arm port/id
- `scripts/configs/data_collection.yaml`: port, camera, task, episode 수, 안전 제한
- `scripts/configs/training.yaml`: dataset, step, batch, model profile
- `scripts/configs/models/smolvla_lora.yaml`: pretrained model과 LoRA rank/alpha
- `scripts/configs/inference.yaml`: checkpoint, 실행 시간, 안전 제한
- `scripts/configs/profiling.yaml`: offline 측정 횟수와 Nsight 옵션

하드웨어를 열지 않고 전체 설정·경로·shell/Python 코드를 검사하려면:

```bash
./scripts/test_project.bash
```

## 3. 실행 순서

팔 전원을 켜기 전 leader/follower를 비슷한 중립 자세에 둔다. 이상 동작 시 즉시 `Ctrl+C`
또는 모터 전원을 차단한다.

### Calibration

```bash
./launchfiles/calibrate.bash
```

기본값은 follower와 leader를 차례로 calibration한다. 중앙 자세에서 Enter를 누르고 안내에
따라 모든 관절과 gripper를 최소/최대 범위까지 천천히 움직인다.

### 데이터 수집

```bash
./launchfiles/collect_data.bash
```

키보드 동작은 다음과 같다.

- `Enter`: 첫 episode 시작
- `→`: 현재 episode 종료. reset 중 다시 누르면 다음 episode 시작
- `←`: 현재 episode 폐기 후 재수집
- `Esc`: 전체 수집 종료 및 dataset finalize

각 결과는 `data/collected_datasets/<run>/dataset`에 생성되고 성공한 최신 run은
`data/collected_datasets/latest`로 연결된다. 주요 dataset 내부 구조는 다음과 같다.

```text
dataset/
  data/chunk-000/file-000.parquet
  videos/observation.images.wrist/chunk-000/file-000.mp4
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
```

`max_relative_target`을 넘는 target은 follower에 보내기 전 관절별로 clamp되고 WARNING이
`artifacts/console.log`에 남는다. 이 저장소의 작은 LeRobot patch는 raw leader target 대신
실제로 clamp되어 전송된 target을 dataset action으로 저장한다. clamp가 반복된 episode는
calibration/동작을 확인하고 다시 수집하는 편이 좋다.

### SmolVLA LoRA 학습

```bash
./launchfiles/train.bash
```

`lerobot/smolvla_base`는 처음 실행 때 `data/models/huggingface`로 자동 다운로드된다. checkpoint와
학습 console log는 `data/training_outputs/<run>`에 저장된다. 모델 구조 실험은
`scripts/configs/models/smolvla_lora.yaml`을 복사하고 `training.yaml`의 `model_config`만
바꾼다.

### 실시간 inference

```bash
./launchfiles/inference.bash
```

기본 checkpoint는 최신 학습 결과다. `max_relative_target`은 tick당 최대 관절 변화(degree),
`interpolation_multiplier`는 action 사이 보간 횟수다. 각 control decision latency는
`latency.jsonl`, 평균/p50/p95/p99는 `latency_summary.json`에 저장된다.

### CUDA/Nsight 측정

```bash
./launchfiles/benchmark_inference.bash
./launchfiles/profile_nsys.bash
./launchfiles/profile_ncu.bash
```

세 명령은 실제 dataset frame과 checkpoint로 매회 실제 policy forward를 수행하지만 로봇에는
명령하지 않는다. 결과는 각각 JSON, `.nsys-rep`, `.ncu-rep`다. Nsight Compute는 kernel
replay로 매우 느리므로 live arm 제어에 직접 붙이지 않는다. profiler 명령에는 JetPack의
`nsys`와 `ncu`가 `PATH`에 설치되어 있어야 한다.

모든 run의 `artifacts/`에는 원본 YAML, 해석된 YAML, 실행 명령, console log, Git revision,
OS/Python/Torch/CUDA/GPU, Nsight version이 자동 기록된다.

## Git과 기존 ROS2 코드

모델·dataset·checkpoint·log·calibration·venv·LeRobot clone은 `.gitignore`로 제외되어 GitHub
repo가 커지지 않는다. 추적 여부는 `git status --short --ignored`로 확인한다.

기존 rosbag joint/image record/replay 패키지는
[`libraries/ros2_bridge`](libraries/ros2_bridge/README.md)에 보존했다. ROS 2
Jazzy와 colcon이 설치된 환경에서만 다음처럼 별도로 build한다.

```bash
BUILD_ROS2=1 ./scripts/test_project.bash
```
