"""Small model-profile registry used to make architecture experiments explicit."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_utils import load_yaml, require_keys  # noqa: E402


def load_model_profile(path: str | Path) -> dict:
    profile = load_yaml(path)
    require_keys(profile, "policy", "peft", context="model profile")
    require_keys(profile["policy"], "path", context="model profile.policy")
    if str(profile["peft"].get("method_type", "")).upper() != "LORA":
        raise ValueError("이 프로젝트의 기본 fine-tuning 경로는 peft.method_type: LORA입니다.")
    return profile
