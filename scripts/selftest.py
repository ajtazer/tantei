"""Synthetic end-to-end check: no real footage needed.

1. Renders a sharp "ground truth" image with small text.
2. Simulates a shaky, low-res, noisy, JPEG-compressed clip of it.
3. Runs `stack` on the clip and reports PSNR vs a single frame.
4. Cuts sliding windows from a wide scene and runs `mosaic`.

    python scripts/selftest.py [workdir]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tantei.stack import stack  # noqa: E402
from tantei.mosaic import mosaic  # noqa: E402

rng = np.random.default_rng(7)
work = Path(sys.argv[1] if len(sys.argv) > 1 else "work/selftest")
(work / "clip").mkdir(parents=True, exist_ok=True)
(work / "pan").mkdir(parents=True, exist_ok=True)

# --- ground truth: text card on a textured background (4x resolution)
S = 4
truth = np.full((200 * S, 320 * S, 3), 235, np.uint8)
truth[:] = cv2.GaussianBlur(rng.integers(150, 255, truth.shape, dtype=np.uint8), (0, 0), 6)
cv2.rectangle(truth, (40 * S, 50 * S), (280 * S, 150 * S), (40, 30, 30), -1)
for i, line in enumerate(["TANTEI SELFTEST 0417", "PLATE KA-05-MX-2231", "stack > single frame"]):
    cv2.putText(truth, line, (52 * S, (72 + 11 * i) * S), cv2.FONT_HERSHEY_SIMPLEX, 0.27 * S,
                (240, 240, 240), max(1, S // 2), cv2.LINE_AA)

# --- degrade: random sub-pixel shake + small rotation, downsample, noise, JPEG
frames = []
for k in range(16):
    dx, dy, ang = rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-1.5, 1.5)
    M = cv2.getRotationMatrix2D((truth.shape[1] / 2, truth.shape[0] / 2), ang, 1.0)
    M[:, 2] += (dx * S, dy * S)
    moved = cv2.warpAffine(truth, M, (truth.shape[1], truth.shape[0]), borderMode=cv2.BORDER_REFLECT)
    low = cv2.resize(cv2.GaussianBlur(moved, (0, 0), S * 0.5), None, fx=1 / S, fy=1 / S, interpolation=cv2.INTER_AREA)
    low = np.clip(low + rng.normal(0, 6, low.shape), 0, 255).astype(np.uint8)
    _, jpg = cv2.imencode(".jpg", low, [cv2.IMWRITE_JPEG_QUALITY, 55])
    f = work / "clip" / f"frame_{k + 1:04d}.png"
    cv2.imwrite(str(f), cv2.imdecode(jpg, 1))
    frames.append(f)

roi = (40, 50, 240, 100)
ref = 0
merged, report = stack(frames, roi, ref=ref, scale=S, min_score=0.6)
single = cv2.resize(cv2.imread(str(frames[ref]))[50:150, 40:280], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)


def psnr_vs_single(img):
    # align each result to the truth crop before scoring (ref frame was shaken too)
    t = cv2.cvtColor(truth[50 * S:150 * S, 40 * S:280 * S], cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    w = np.eye(2, 3, dtype=np.float32)
    try:
        _, w = cv2.findTransformECC(t, g, w, cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        pass
    a = cv2.warpAffine(g, w, (t.shape[1], t.shape[0]), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    m = 12 * S  # ignore borders
    return cv2.PSNR(t[m:-m, m:-m], a[m:-m, m:-m])


p1, p2 = psnr_vs_single(single), psnr_vs_single(merged)
cv2.imwrite(str(work / "single_bicubic.png"), single)
cv2.imwrite(str(work / "stacked.png"), merged)
kept = sum(r["kept"] for r in report)
print(f"stack: kept {kept}/{len(frames)} frames | PSNR single {p1:.2f} dB -> stacked {p2:.2f} dB")

# --- mosaic: slide a window across a wide synthetic scene
scene = cv2.GaussianBlur(rng.integers(0, 255, (360, 1400, 3), dtype=np.uint8), (0, 0), 3)
for x in range(60, 1400, 110):
    cv2.rectangle(scene, (x, int(rng.integers(40, 200))), (x + 50, int(rng.integers(220, 330))),
                  tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    cv2.putText(scene, str(x), (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
pan = []
for i, x in enumerate(range(0, 1400 - 480 + 1, 60)):
    f = work / "pan" / f"frame_{i + 1:04d}.png"
    cv2.imwrite(str(f), scene[:, x:x + 480])
    pan.append(f)
pano = mosaic(pan, mode="scans", step=2, max_width=0)
cv2.imwrite(str(work / "mosaic.png"), pano)
print(f"mosaic: {len(pan)} frames of 480px -> {pano.shape[1]}x{pano.shape[0]} (scene is 1400x360)")

ok = p2 > p1 and pano.shape[1] > 1000
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
