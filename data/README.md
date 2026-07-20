# Runtime data

이 폴더의 큰 파일은 모두 `.gitignore` 대상이다.

- `collected_datasets/<run>/dataset`: 직접 수집한 LeRobotDataset v3
- `downloaded_datasets`: Hugging Face에서 받은 데이터 cache
- `models`: Hugging Face 모델 cache/weight
- `training_outputs`: checkpoint와 학습 로그
- `inference_logs`: 실시간 latency, benchmark, Nsight report
- `calibration`: follower/leader calibration JSON
- `calibration_runs`: calibration 당시 YAML과 console log
- `rosbags`: 이전 ROS2 bridge용 bag

각 실행 결과의 `artifacts/`에는 사용한 원본 YAML, 완전히 해석된 YAML, 명령어,
console log, Git/CUDA/하드웨어 정보가 함께 저장된다.
