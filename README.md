<p align="center">
  <img src="https://media1.tenor.com/m/VXDCRP3Lr_oAAAAC/chika-detective.gif" alt="Detective Chika, the tantei mascot" width="360">
  <br>
  <sub>Mascot: Detective Chika Fujiwara (<i>Kaguya-sama: Love is War</i>), via <a href="https://tenor.com/view/chika-detective-kaguya-sama-love-gif-23662119">Tenor</a>. Character © Aka Akasaka / Shueisha / A-1 Pictures.</sub>
</p>

# tantei 探偵

**Get more truth out of bad footage.**

tantei is a small command-line kit for OSINT and investigations. It takes blurry, low-resolution, compressed video (phone clips, CCTV, re-shared social media) and pulls out as much real detail as the pixels actually hold:

- **Find the moment:** dump frames and scan them on a numbered contact sheet.
- **Stack frames:** track an object across frames, align it to within a fraction of a pixel and merge. You get detail and cleanliness that no single frame has.
- **Stitch the bigger picture:** join overlapping frames from a panning camera into one wide view of the scene.
- **Enhance honestly:** crop, denoise, boost local contrast and sharpen, all without inventing anything.
- **AI upscale:** run 4x-UltraSharp, Real-ESRGAN or any model spandrel can load, clearly marked as generative.
- **Compare:** put every result side by side with the raw pixels, so nobody mistakes model guesses for evidence.

```bash
pip install -r requirements.txt
python -m tantei extract clip.mp4 --start 12 --end 13 -o frames
python -m tantei sheet frames -o sheet.png
python -m tantei stack frames --roi 220,140,90,60 --ref 9 -o stacked.png
```

The full method is in **[PLAYBOOK.md](PLAYBOOK.md)**: which step to use when, how to pick a region, and how to report findings without overclaiming.

> **The one rule:** stacking and classic enhancement recover real signal. AI upscalers make up plausible detail. Never read text, numbers or faces off an AI-upscaled image unless the stacked or raw version supports it.

## Roadmap

- [ ] **Auto ROI tracking:** click once and follow the object through the clip (CSRT/SAM2), with no hand-typed coordinates.
- [ ] **Deblur:** estimate the motion-blur kernel from a streak, then deconvolve (Wiener / Richardson–Lucy).
- [ ] **Burst super-resolution models** (e.g. BSRT, BIPNet) as a learned upgrade over mean stacking.
- [ ] **OCR with confidence:** run OCR on raw, stacked and enhanced versions, and flag characters that only appear after AI upscaling.
- [ ] **Face and plate modules** with built-in "hallucination risk" overlays.
- [ ] **Metadata and provenance:** EXIF/container dump, re-encode detection, hashes of every intermediate (chain of custody).
- [ ] **Reflection and screen analysis:** perspective-correct a tilted phone or monitor screen, then stack it.
- [ ] **Video panorama with moving objects:** background mosaic plus object layers (a "where was everyone" frame).
- [ ] **Geolocation helpers:** skyline/landmark crops ready for reverse-image search.
- [ ] **Web UI:** drag in a clip, draw a box, get a report.
- [ ] **One-page case report:** before/after panels, method log and confidence notes, exported to PDF.

## License

MIT
