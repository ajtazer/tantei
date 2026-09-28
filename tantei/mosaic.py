"""Stitch overlapping frames from a panning / moving camera into one wide view
("the bigger picture" no single frame shows)."""
import cv2


def mosaic(frames, mode="panorama", step=1, max_width=1600):
    ims = [cv2.imread(str(p)) for p in frames[::step]]
    ims = [i for i in ims if i is not None]
    if len(ims) < 2:
        raise SystemExit("need at least 2 frames")
    st = cv2.Stitcher_create(cv2.Stitcher_SCANS if mode == "scans" else cv2.Stitcher_PANORAMA)
    status, pano = st.stitch(ims)
    if status != cv2.Stitcher_OK:
        names = {1: "not enough overlap/features", 2: "homography estimation failed",
                 3: "camera parameter adjustment failed"}
        raise SystemExit(f"stitch failed: {names.get(status, status)} - try --step 2..5 or --mode scans")
    if pano.shape[1] > max_width > 0:
        pano = cv2.resize(pano, (max_width, int(pano.shape[0] * max_width / pano.shape[1])),
                          interpolation=cv2.INTER_AREA)
    return pano
