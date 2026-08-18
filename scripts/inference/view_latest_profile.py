#!/usr/bin/env python3
"""Open the newest local/received profile in its desktop GUI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_utils import absolute_path  # noqa: E402
from system_config import load_system  # noqa: E402


PATTERNS = {
    "ncu": "*.ncu-rep",
    "nsys": "*.nsys-rep",
    "torch": "torch_trace*.json",
}


def newest_profile(roots: list[Path], kind: str) -> tuple[str, Path]:
    kinds = PATTERNS if kind == "auto" else {kind: PATTERNS[kind]}
    matches: list[tuple[float, str, Path]] = []
    for selected, pattern in kinds.items():
        for root in roots:
            if root.is_dir():
                for path in root.rglob(pattern):
                    if path.is_file():
                        matches.append((path.stat().st_mtime, selected, path.resolve()))
    if not matches:
        raise FileNotFoundError(f"{kind} profile을 찾지 못했습니다: {', '.join(map(str, roots))}")
    _mtime, selected, path = max(matches, key=lambda item: item[0])
    return selected, path


def executable(*names: str) -> str:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(f"시각화 실행 파일이 없습니다: {', '.join(names)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("kind", nargs="?", choices=("auto", "ncu", "nsys", "torch"), default="auto")
    parser.add_argument("--print-only", action="store_true", help="GUI를 열지 않고 선택 결과만 출력")
    args = parser.parse_args()
    config = load_system(args.config)
    roots = [
        Path(absolute_path(config["transfer"].get("profiling_root", "data/profiling_from_orin"))),
        Path(absolute_path(config["runs"]["benchmark"]["output_root"])),
    ]
    kind, path = newest_profile(roots, args.kind)
    print(f"latest {kind} profile: {path}")
    if args.print_only:
        return 0
    if kind == "ncu":
        command = [executable("ncu-ui", "nv-nsight-cu"), str(path)]
    elif kind == "nsys":
        command = [executable("nsys-ui", "nsight-sys"), str(path)]
    else:
        # Perfetto imports PyTorch Chrome traces. The selected path is printed so
        # it can be dropped into the opened UI without searching the repository.
        browser = executable(
            "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "xdg-open"
        )
        command = [browser, "https://ui.perfetto.dev"]
        print(f"Perfetto에서 Open trace file을 눌러 이 파일을 여세요: {path}")
    subprocess.Popen(command, start_new_session=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
