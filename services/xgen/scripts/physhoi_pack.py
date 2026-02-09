from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import yaml

from xgen.pose.physhoi_motion import load_physhoi_motion, validate_physhoi_motion, summarize_physhoi_motion


def build_yaml(motion_files: List[Path], weights: List[float]) -> dict:
    motions = []
    for path, weight in zip(motion_files, weights):
        motions.append({"file": path.as_posix(), "weight": float(weight)})
    return {"motions": motions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack PhysHOI motions and emit YAML list")
    parser.add_argument("--motions", nargs="+", required=True, help="List of .pt motion files")
    parser.add_argument("--weights", nargs="+", type=float, help="Weights for each motion")
    parser.add_argument("--output", required=True, help="YAML output path")
    parser.add_argument("--validate", action="store_true", help="Validate motions before writing YAML")
    args = parser.parse_args()

    motion_paths = [Path(p) for p in args.motions]
    weights = args.weights or [1.0 for _ in motion_paths]
    if len(weights) != len(motion_paths):
        raise SystemExit("weights count must match motions count")

    summaries = []
    if args.validate:
        for path in motion_paths:
            tensor = load_physhoi_motion(path)
            validate_physhoi_motion(tensor)
            summaries.append({"file": path.as_posix(), **summarize_physhoi_motion(tensor)})

    payload = build_yaml(motion_paths, weights)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    print(f"Wrote {output_path}")
    if summaries:
        print("Summaries:")
        for summary in summaries:
            print(summary)


if __name__ == "__main__":
    main()
