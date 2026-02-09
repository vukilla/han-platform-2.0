from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def render_preview(source_video: str | Path, output_path: str | Path, overlay_text: str | None = None) -> Path:
    """Render preview clip using FFmpeg (fallback to copy if FFmpeg missing)."""
    source = Path(source_video)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        output.write_bytes(b"")
        return output

    cmd = ["ffmpeg", "-y", "-i", str(source)]
    if overlay_text:
        cmd += [
            "-vf",
            f"drawtext=text='{overlay_text}':fontcolor=white:fontsize=24:x=10:y=10:box=1:boxcolor=black@0.4",
        ]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", str(output)]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        shutil.copyfile(source, output)
    return output
