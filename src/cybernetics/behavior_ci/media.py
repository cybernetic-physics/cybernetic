"""mp4 -> animated GIF transcode for inline PR-comment replays.

GitHub renders animated GIFs inline in PR/issue comments (proxied through camo) but never
plays an mp4 from a URL or repo path, so the hosted Isaac replay mp4 is transcoded here to a
compact, camo-safe looping GIF. Uses the static ffmpeg bundled by ``imageio-ffmpeg`` (no
system ffmpeg required). Best-effort: callers swallow failures so a missing GIF never
invalidates the (authoritative) mp4 bundle.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# camo refuses/truncates very large animated GIFs even when the origin serves 200; keep well
# under that. The 720p/24fps mp4 typically yields 0.8-2.5MB at 480px/12fps/128 colors.
DEFAULT_SIZE_CAP = 5_000_000


def looks_like_gif(data: bytes) -> bool:
    return len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a")


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _transcode(ffmpeg: str, mp4_path: str, gif_path: str, fps: int, width: int) -> None:
    """Two-pass palettegen -> high-quality small GIF, infinite loop."""
    palette = gif_path + ".palette.png"
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            mp4_path,
            "-vf",
            scale + ",palettegen=max_colors=128:stats_mode=diff",
            palette,
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            mp4_path,
            "-i",
            palette,
            "-lavfi",
            scale + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            "-loop",
            "0",
            gif_path,
        ],
        check=True,
        capture_output=True,
    )
    try:
        os.unlink(palette)
    except OSError:
        pass


def mp4_to_gif(
    mp4: bytes, *, fps: int = 12, width: int = 480, size_cap: int = DEFAULT_SIZE_CAP
) -> Optional[bytes]:
    """Transcode mp4 bytes to a looping GIF.

    Returns the GIF bytes, or ``None`` if the input is empty/invalid or the result stays
    over ``size_cap`` even after a smaller retry (so an oversized GIF is never emitted and
    camo never refuses it). Raises if ffmpeg itself fails — callers run this best-effort.
    """
    if not mp4:
        return None
    ffmpeg = _ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        mp4_path = os.path.join(td, "in.mp4")
        gif_path = os.path.join(td, "out.gif")
        Path(mp4_path).write_bytes(mp4)
        for f, w in ((fps, width), (10, 400)):
            _transcode(ffmpeg, mp4_path, gif_path, f, w)
            data = Path(gif_path).read_bytes()
            if looks_like_gif(data) and len(data) <= size_cap:
                return data
        return None
