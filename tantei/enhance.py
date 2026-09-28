"""Classic, non-generative enhancement. Nothing here invents pixels."""
import cv2
import numpy as np


def crop(im, roi):
    x, y, w, h = roi
    return im[y:y + h, x:x + w]


def zoom(im, factor, nearest=False):
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
    return cv2.resize(im, None, fx=factor, fy=factor, interpolation=interp)


def unsharp(im, amount=0.8, radius=3.0):
    f = im.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), radius)
    return cv2.addWeighted(f, 1 + amount, blur, -amount, 0).clip(0, 255).astype(np.uint8)


def clahe(im, clip=3.0, grid=4):
    """Local contrast boost on luminance; good for faint text and dark regions."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clip, (grid, grid)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(im, strength=5):
    return cv2.fastNlMeansDenoisingColored(im, None, strength, strength, 7, 21)


def gray(im):
    return cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
