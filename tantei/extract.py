"""Pull frames out of a video and build contact sheets for quick scanning."""
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


def probe(video):
    """Prefer ffprobe (handles HEVC/phone codecs pip OpenCV can't decode)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_packets:format=duration",
             "-of", "default=nw=1", str(video)],
            check=True, capture_output=True, text=True)
        kv = dict(line.split("=", 1) for line in r.stdout.split())
        num, den = kv["r_frame_rate"].split("/")
        return {"codec": kv["codec_name"], "fps": int(num) / int(den), "frames": int(kv["nb_read_packets"]),
                "width": int(kv["width"]), "height": int(kv["height"]), "duration": float(kv["duration"])}
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, ValueError):
        pass
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0
    cap.release()
    return info


def extract(video, out_dir, start=0.0, end=None, every=1):
    """Write every `every`-th frame between start and end (seconds) as lossless PNG.

    Uses ffmpeg when available (accurate seek, no re-encode loss), else OpenCV.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start)]
        if end is not None:
            cmd += ["-t", str(end - start)]
        cmd += ["-i", str(video), "-vsync", "0"]
        if every > 1:
            cmd += ["-vf", f"select=not(mod(n\\,{every}))"]
        cmd += [str(out / "frame_%04d.png")]
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        _extract_cv(video, out, start, end, every)
    files = sorted(out.glob("frame_*.png"))
    print(f"{len(files)} frames -> {out}")
    return files


def _extract_cv(video, out, start, end, every):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    n = k = 0
    while True:
        ok, f = cap.read()
        if not ok or (end is not None and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > end + 1 / fps):
            break
        if n % every == 0:
            k += 1
            cv2.imwrite(str(out / f"frame_{k:04d}.png"), f)
        n += 1
    cap.release()


def contact_sheet(images, out, cols=8, thumb_w=200, label=True):
    """Tile images into one numbered grid so you can pick frames by eye."""
    ims = [cv2.imread(str(p)) for p in images]
    ims = [i for i in ims if i is not None]
    if not ims:
        raise SystemExit("no readable images")
    h = int(ims[0].shape[0] * thumb_w / ims[0].shape[1])
    tiles = []
    for idx, im in enumerate(ims, 1):
        t = cv2.resize(im, (thumb_w, h), interpolation=cv2.INTER_AREA)
        if label:
            cv2.rectangle(t, (0, 0), (44, 22), (0, 0, 0), -1)
            cv2.putText(t, str(idx), (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    rows = math.ceil(len(tiles) / cols)
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (rows * cols - len(tiles))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(out), grid)
    print(f"contact sheet ({len(ims)} frames) -> {out}")
