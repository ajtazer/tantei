# tantei playbook

A field method for pulling reliable detail out of low-quality video. The steps go from **most trustworthy** to **least**. Report findings from the highest step that supports them.

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch/spandrel only needed for step 5
./scripts/fetch_models.sh                # optional: 4x-UltraSharp, RealESRGAN_x4plus
python scripts/selftest.py               # synthetic check, no real footage needed
```

`ffmpeg` on PATH is strongly recommended. Phone footage is often HEVC, which pip-installed OpenCV can't decode.

## 1. Preserve and probe

- Work on a **copy**. Hash the original (`shasum -a 256 clip.mp4`) and write the hash down.
- Always get the **highest-quality source** you can. The original upload beats a screen recording, which beats a re-share. Going from 480p to 1080p doubles every dimension, which is more than any algorithm can recover.

```bash
python -m tantei probe clip.mp4          # codec, fps, frame count, resolution
```

## 2. Find the moment

```bash
python -m tantei extract clip.mp4 --every 6 -o scan         # whole clip, sparse
python -m tantei sheet scan --cols 10 -o scan.png           # eyeball it
python -m tantei extract clip.mp4 --start 12 --end 13 -o frames   # every frame in the window
python -m tantei sheet frames -o frames.png
```

The contact sheet is numbered, so you can refer to frames by index: `--ref 9` means the 9th frame in the folder.

## 3. Look at the raw pixels first

```bash
python -m tantei enhance frames/frame_0009.png --roi 200,120,140,100 --zoom 6 --nearest -o raw_zoom.png
```

Nearest-neighbour zoom shows exactly how many pixels your target spans. As a rough guide:

| Target height in source | What's realistic |
|---|---|
| ≥ 12 px per text line | readable from a single frame |
| 6–12 px | often readable after stacking |
| 3–6 px | word shapes and layout only; letters rarely |
| < 3 px | presence only |

## 4. Stack (multi-frame super-resolution)

The core technique. Pick a reference frame where the target is sharp and fully in frame. Then choose an ROI box `x,y,w,h` around it, with a little margin, from the raw zoom.

```bash
python -m tantei stack frames --roi 220,140,90,60 --ref 9 --scale 4 -o stacked.png
```

- Every frame is located near the ROI (cross-correlation), then aligned sub-pixel with ECC and merged.
- `stacked.tsv` lists per-frame scores. Frames below `--min-score` (default 0.9) are dropped as motion-blurred, occluded or out of frame.
- `--motion homography` for flat things seen at changing angles (screens, documents, signs). Use `translation` when little moves.
- `--method median` resists occluders passing through (a hand, a car).
- More good frames means a cleaner result. 6–15 well-aligned frames usually beats 40 sloppy ones.

## 5. Enhance (non-generative)

```bash
python -m tantei enhance stacked.png --clahe 3 --sharpen 0.8 -o stacked_enh.png
python -m tantei enhance stacked.png --gray --clahe 4 --zoom 2 -o text_pass.png   # faint text
```

CLAHE brings out faint text and dark regions. Denoise (`--denoise 5`) helps single frames. It's rarely needed after stacking.

## 6. AI upscale (generative, viewing aid only)

```bash
python -m tantei upscale stacked.png -m models/4x-UltraSharp.pth -o up
python -m tantei upscale stacked.png -m models/RealESRGAN_x4plus.pth -o up
```

- **4x-UltraSharp** is conservative. It sharpens edges and rarely invents structure.
- **Real-ESRGAN x4plus** is aggressive. It turns noise into crisp shapes, including **fake letters**.
- Diffusion restorers (SUPIR and similar) go further still. They produce beautiful and confidently wrong fine detail, and need 12 GB+ of GPU memory.
- Use the lower `--tile` values on 8 GB machines.

**Never** treat characters, digits or facial features that appear *only* in an AI-upscaled output as findings.

## 7. Stitch the bigger picture

When the camera pans, no single frame shows the whole scene. A mosaic does.

```bash
python -m tantei extract clip.mp4 --every 3 -o pan
python -m tantei mosaic pan --step 2 -o scene.png              # rotating/handheld camera
python -m tantei mosaic pan --step 2 --mode scans -o scene.png # sliding / flat scene
```

If stitching fails, try a larger `--step` (more motion between frames) or `--mode scans`. Moving people will ghost. Stitch a short window where the scene is static.

## 8. Compare and report

```bash
python -m tantei compare raw_zoom.png stacked.png stacked_enh.png up/stacked_4x-UltraSharp.png \
  --labels raw,stacked,enhanced,ai-upscale -o panel.png
```

For each finding, write:
1. **What** you observed.
2. **Which step** first shows it (raw / stacked / enhanced / AI).
3. **Confidence:** high if raw or stacked shows it, low if it appears only after AI upscaling.
4. **Frames used**, with the `.tsv` attached, plus the source hash.

## Worked pattern: "what's on that screen?"

1. Take the sheet. Find frames where the screen is steady and unobstructed.
2. Run a raw zoom to measure the text height in pixels (see the table in step 3).
3. Stack with `--motion homography --method median`.
4. Do a `--gray --clahe 4` pass.
5. If the lines are still under ~5 px, report layout only, e.g. "sign with two lines of text, layout visible, characters unreadable". Ask for a better source instead of upscaling harder.

## Ethics

Use this on material you're authorised to analyse. Blur faces and personal data of bystanders before you share outputs, and keep the chain of custody. Enhancement increases how much a viewer trusts an image faster than it increases how accurate the image is, so report honestly.
