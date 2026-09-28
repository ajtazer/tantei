"""tantei: get more truth out of low-quality footage."""
import argparse
import glob
from pathlib import Path

import cv2


def _roi(s):
    try:
        x, y, w, h = (int(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    return x, y, w, h


def _frames(src):
    p = Path(src)
    files = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg")) if p.is_dir() else sorted(Path(f) for f in glob.glob(src))
    if not files:
        raise SystemExit(f"no frames found at {src}")
    return files


def cmd_probe(a):
    from .extract import probe
    for k, v in probe(a.video).items():
        print(f"{k:9} {v:.3f}" if isinstance(v, float) else f"{k:9} {v}")


def cmd_extract(a):
    from .extract import extract
    extract(a.video, a.out, a.start, a.end, a.every)


def cmd_sheet(a):
    from .extract import contact_sheet
    contact_sheet(_frames(a.frames), a.out, a.cols, a.width)


def cmd_stack(a):
    from .stack import stack, save_report
    frames = _frames(a.frames)
    ref = a.ref - 1 if a.ref else None
    merged, report = stack(frames, a.roi, ref, a.scale, a.motion, a.search, a.min_score, a.method)
    cv2.imwrite(a.out, merged)
    save_report(report, Path(a.out).with_suffix(".tsv"))
    kept = sum(r["kept"] for r in report)
    print(f"merged {kept}/{len(report)} frames at {a.scale}x -> {a.out} (per-frame scores in .tsv)")


def cmd_enhance(a):
    from . import enhance as e
    im = cv2.imread(a.image)
    if a.roi:
        im = e.crop(im, a.roi)
    if a.denoise:
        im = e.denoise(im, a.denoise)
    if a.clahe:
        im = e.clahe(im, a.clahe)
    if a.gray:
        im = e.gray(im)
    if a.zoom > 1:
        im = e.zoom(im, a.zoom, a.nearest)
    if a.sharpen:
        im = e.unsharp(im, a.sharpen)
    cv2.imwrite(a.out, im)
    print(f"-> {a.out} {im.shape[1]}x{im.shape[0]}")


def cmd_upscale(a):
    from .upscale import load, upscale
    model, dev = load(a.model)
    print(f"{model.architecture.name} x{model.scale} on {dev}")
    for src in a.images:
        out = Path(a.out) / (Path(src).stem + f"_{Path(a.model).stem}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), upscale(cv2.imread(src), model, dev, a.tile))
        print(f"-> {out}")


def cmd_mosaic(a):
    from .mosaic import mosaic
    cv2.imwrite(a.out, mosaic(_frames(a.frames), a.mode, a.step, a.max_width))
    print(f"-> {a.out}")


def cmd_compare(a):
    from .compare import side_by_side
    labels = a.labels.split(",") if a.labels else None
    cv2.imwrite(a.out, side_by_side(a.images, a.height, labels))
    print(f"-> {a.out}")


def main():
    p = argparse.ArgumentParser(prog="tantei", description=__doc__)
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("probe", help="video metadata")
    s.add_argument("video")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("extract", help="dump frames from a time window as PNG")
    s.add_argument("video")
    s.add_argument("-o", "--out", default="frames")
    s.add_argument("--start", type=float, default=0.0, help="seconds")
    s.add_argument("--end", type=float, help="seconds")
    s.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("sheet", help="numbered contact sheet of frames")
    s.add_argument("frames", help="folder or glob")
    s.add_argument("-o", "--out", default="sheet.png")
    s.add_argument("--cols", type=int, default=8)
    s.add_argument("--width", type=int, default=200, help="thumbnail width")
    s.set_defaults(fn=cmd_sheet)

    s = sub.add_parser("stack", help="multi-frame super-resolution of a region")
    s.add_argument("frames", help="folder or glob")
    s.add_argument("--roi", type=_roi, required=True, help="x,y,w,h in the reference frame")
    s.add_argument("--ref", type=int, help="1-based reference frame (default: middle)")
    s.add_argument("--scale", type=int, default=4)
    s.add_argument("--motion", choices=["translation", "euclidean", "affine", "homography"], default="affine")
    s.add_argument("--search", type=int, default=80, help="px to search around ROI per frame")
    s.add_argument("--min-score", type=float, default=0.9, help="ECC gate, 0-1")
    s.add_argument("--method", choices=["mean", "median"], default="mean")
    s.add_argument("-o", "--out", default="stacked.png")
    s.set_defaults(fn=cmd_stack)

    s = sub.add_parser("enhance", help="crop / denoise / CLAHE / zoom / sharpen (non-generative)")
    s.add_argument("image")
    s.add_argument("--roi", type=_roi)
    s.add_argument("--denoise", type=int, default=0)
    s.add_argument("--clahe", type=float, default=0, help="clip limit, e.g. 3")
    s.add_argument("--gray", action="store_true")
    s.add_argument("--zoom", type=float, default=1)
    s.add_argument("--nearest", action="store_true", help="zoom without interpolation (see raw pixels)")
    s.add_argument("--sharpen", type=float, default=0, help="unsharp amount, e.g. 0.8")
    s.add_argument("-o", "--out", default="enhanced.png")
    s.set_defaults(fn=cmd_enhance)

    s = sub.add_parser("upscale", help="AI upscale with any spandrel-supported model (generative!)")
    s.add_argument("images", nargs="+")
    s.add_argument("-m", "--model", required=True, help=".pth/.safetensors, see scripts/fetch_models.sh")
    s.add_argument("--tile", type=int, default=256)
    s.add_argument("-o", "--out", default="upscaled")
    s.set_defaults(fn=cmd_upscale)

    s = sub.add_parser("mosaic", help="stitch overlapping frames into one wide view")
    s.add_argument("frames", help="folder or glob")
    s.add_argument("--mode", choices=["panorama", "scans"], default="panorama")
    s.add_argument("--step", type=int, default=3, help="use every Nth frame")
    s.add_argument("--max-width", type=int, default=2400)
    s.add_argument("-o", "--out", default="mosaic.png")
    s.set_defaults(fn=cmd_mosaic)

    s = sub.add_parser("compare", help="side-by-side panel of images")
    s.add_argument("images", nargs="+")
    s.add_argument("--labels", help="comma-separated")
    s.add_argument("--height", type=int, default=900)
    s.add_argument("-o", "--out", default="compare.png")
    s.set_defaults(fn=cmd_compare)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
