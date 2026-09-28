"""Multi-frame super-resolution: track a region across frames, align it
sub-pixel, and merge. Each frame samples the scene at a slightly different
offset, so averaging aligned frames recovers detail and kills compression noise
that no single frame has."""
from pathlib import Path

import cv2
import numpy as np


def _gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def _locate(frame, template, center, search):
    """Coarse: find the template near `center` with normalized cross-correlation."""
    th, tw = template.shape[:2]
    cx, cy = center
    x0, y0 = max(cx - search, 0), max(cy - search, 0)
    x1 = min(cx + tw + search, frame.shape[1])
    y1 = min(cy + th + search, frame.shape[0])
    win = _gray(frame[y0:y1, x0:x1])
    if win.shape[0] < th or win.shape[1] < tw:
        return None, 0.0
    r = cv2.matchTemplate(win, _gray(template), cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(r)
    return (x0 + loc[0], y0 + loc[1]), score


def stack(frames, roi, ref=None, scale=4, motion="affine", search=80,
          min_score=0.9, method="mean"):
    """Align `roi` (x, y, w, h in the reference frame) across frames and merge.

    Returns (merged image at `scale`x, list of per-frame dicts with scores).
    Frames whose ECC correlation < min_score are rejected (motion blur,
    occlusion, the object leaving frame).
    """
    ims = [cv2.imread(str(p)) for p in frames]
    ref = ref if ref is not None else len(ims) // 2
    x, y, w, h = roi
    ref_img = ims[ref]
    template = ref_img[y:y + h, x:x + w]
    ref_up = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ref_g = _gray(ref_up).astype(np.float32)
    mode = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE, "homography": cv2.MOTION_HOMOGRAPHY}[motion]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    kept, report = [], []
    pad = search + max(w, h)
    for i, im in enumerate(ims):
        loc, ncc = _locate(im, template, (x, y), search)
        if loc is None:
            report.append({"frame": str(frames[i]), "ncc": 0, "ecc": 0, "kept": False})
            continue
        big = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        patch = big[loc[1] + pad:loc[1] + pad + h, loc[0] + pad:loc[0] + pad + w]
        up = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        warp = np.eye(3, 3, dtype=np.float32) if mode == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
        try:
            ecc, warp = cv2.findTransformECC(ref_g, _gray(up).astype(np.float32), warp, mode, crit, None, 5)
        except cv2.error:
            ecc = 0.0
        size = (ref_up.shape[1], ref_up.shape[0])
        flags = cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP
        if mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(up, warp, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
        else:
            aligned = cv2.warpAffine(up, warp, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
        ok = ecc >= min_score
        if ok:
            kept.append(aligned.astype(np.float32))
        report.append({"frame": str(frames[i]), "ncc": round(float(ncc), 3),
                       "ecc": round(float(ecc), 3), "kept": ok})

    if not kept:
        raise SystemExit("no frames passed min_score; lower --min-score or pick a better ROI/reference")
    st = np.stack(kept)
    merged = np.median(st, 0) if method == "median" else st.mean(0)
    return merged.clip(0, 255).astype(np.uint8), report


def save_report(report, path):
    lines = ["frame\tncc\tecc\tkept"] + [f"{Path(r['frame']).name}\t{r['ncc']}\t{r['ecc']}\t{r['kept']}" for r in report]
    Path(path).write_text("\n".join(lines) + "\n")
