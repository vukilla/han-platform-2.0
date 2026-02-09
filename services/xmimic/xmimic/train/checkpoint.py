from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output)
    return output


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["state_dict"])
