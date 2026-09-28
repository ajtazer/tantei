"""Model-based upscaling via spandrel, which loads most community SR weights
(4x-UltraSharp, RealESRGAN_x4plus, SwinIR, HAT, ...) from a single .pth/.safetensors.

GENERATIVE: these models synthesize plausible texture. Use them for viewing,
never as evidence of fine detail (text, numbers, faces) on their own."""
import cv2
import numpy as np


def _device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_path):
    import torch
    from spandrel import ModelLoader
    dev = _device(torch)
    model = ModelLoader().load_from_file(str(model_path)).to(dev).eval()
    return model, dev


def upscale(im, model, dev, tile=256, overlap=16):
    """Tiled inference so large images fit on 8 GB machines."""
    import torch
    s = model.scale
    h, w = im.shape[:2]
    src = im[:, :, ::-1].astype(np.float32) / 255
    out = np.zeros((h * s, w * s, 3), np.float32)
    weight = np.zeros((h * s, w * s, 1), np.float32)
    step = tile - overlap
    for y in range(0, max(h - overlap, 1), step):
        for x in range(0, max(w - overlap, 1), step):
            y1, x1 = min(y + tile, h), min(x + tile, w)
            y0, x0 = max(y1 - tile, 0), max(x1 - tile, 0)
            t = torch.from_numpy(src[y0:y1, x0:x1].copy()).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                o = model(t)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            out[y0 * s:y1 * s, x0 * s:x1 * s] += o
            weight[y0 * s:y1 * s, x0 * s:x1 * s] += 1
    out /= np.maximum(weight, 1)
    return (out[:, :, ::-1] * 255).round().astype(np.uint8)
