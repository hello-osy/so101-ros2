#!/usr/bin/env python3
"""Interactive SO-101 calibration without the normal runtime configure step."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pprint import pformat
from types import MethodType

import draccus

from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.scripts.lerobot_calibrate import CalibrateConfig
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


def install_write_retries(bus, communication_retries: int) -> None:
    """Apply a small retry budget to individual serial writes during calibration."""
    original_write = bus.write

    def write_with_retries(self, *args, **kwargs):
        requested = int(kwargs.get("num_retry", 0))
        kwargs["num_retry"] = max(requested, communication_retries)
        return original_write(*args, **kwargs)

    bus.write = MethodType(write_with_retries, bus)


def load_config(config_path: str) -> CalibrateConfig:
    with draccus.config_type("yaml"):
        return draccus.parse(CalibrateConfig, config_path, args=[])


def make_device(cfg: CalibrateConfig):
    if isinstance(cfg.device, RobotConfig):
        return make_robot_from_config(cfg.device)
    if isinstance(cfg.device, TeleoperatorConfig):
        return make_teleoperator_from_config(cfg.device)
    raise TypeError(f"지원하지 않는 calibration device 설정입니다: {type(cfg.device).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--communication-retries", type=int, default=3)
    args = parser.parse_args()
    if args.communication_retries < 0 or args.communication_retries > 10:
        parser.error("--communication-retries는 0~10 사이여야 합니다.")

    register_third_party_plugins()
    init_logging()
    cfg = load_config(args.config_path)
    logging.info(pformat(asdict(cfg)))
    device = make_device(cfg)
    bus = device.bus
    install_write_retries(bus, args.communication_retries)

    print(
        f"모터 버스 연결 중: {cfg.device.port} "
        f"(통신 실패 시 최대 {args.communication_retries}회 추가 재시도)",
        flush=True,
    )
    bus.connect()
    try:
        # The official CLI calls device.connect(calibrate=False), which runs
        # configure() and briefly re-enables torque before calibration. Direct
        # bus connection avoids that unnecessary and failure-prone transition.
        device.calibrate()
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
