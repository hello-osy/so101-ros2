"""Low-latency camera preview shared by standalone and teleoperation launchers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TextIO


CameraProcess = tuple[subprocess.Popen, TextIO]


def camera_command(config: dict, camera_name: str) -> list[str]:
    cameras = config["devices"]["cameras"]
    if camera_name not in cameras:
        raise ValueError(f"devices.cameras에 '{camera_name}' 카메라가 없습니다.")
    camera = cameras[camera_name]
    if camera.get("type") != "opencv":
        raise ValueError("저지연 미리보기는 opencv 카메라만 지원합니다.")
    ffplay = shutil.which("ffplay")
    if not ffplay:
        raise FileNotFoundError("ffplay가 없습니다. install_system_dependencies.bash를 실행하세요.")

    source = camera["index_or_path"]
    if isinstance(source, int) or str(source).isdigit():
        source = f"/dev/video{source}"
    input_format = {"MJPG": "mjpeg", "YUYV": "yuyv422"}.get(str(camera.get("fourcc", "")))
    command = [
        ffplay,
        "-loglevel",
        "warning",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-f",
        "v4l2",
    ]
    if input_format:
        command.extend(["-input_format", input_format])
    command.extend(
        [
            "-video_size",
            f"{int(camera['width'])}x{int(camera['height'])}",
            "-framerate",
            str(int(camera["fps"])),
            "-window_title",
            f"SO-101 {camera_name}",
            str(source),
        ]
    )
    return command


def start_camera(config: dict, camera_name: str, log_path: Path) -> CameraProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            camera_command(config, camera_name),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log.close()
        raise
    return process, log


def stop_camera(preview: CameraProcess | None) -> None:
    if preview is None:
        return
    process, log = preview
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    log.close()
