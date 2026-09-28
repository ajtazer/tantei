"""Side-by-side panels, so every enhancement is judged against the raw pixels."""
import cv2
import numpy as np


def side_by_side(images, height=900, labels=None):
    ims = [cv2.imread(str(p)) for p in images]
    out = []
    for i, im in enumerate(ims):
        if im is None:
            raise SystemExit(f"cannot read {images[i]}")
        small = im.shape[0] < height / 3
        w = int(im.shape[1] * height / im.shape[0])
        t = cv2.resize(im, (w, height), interpolation=cv2.INTER_NEAREST if small else cv2.INTER_AREA)
        if labels:
            cv2.rectangle(t, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(t, labels[i], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(t)
    return np.hstack(out)
