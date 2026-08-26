"""
Draw SAM3 concept-segmentation masks ("road", "vehicle", "building") on every
frame of alpha.mp4 and write the annotated frames to a new mp4.

Notes / assumptions (confirmed with the user):
- SAM3SemanticPredictor has no native video mode, so this script manually
  loops over every frame of alpha.mp4, calling `predictor.set_image(frame)`
  + `predictor(text=[...])` per frame. This re-runs the SAM3 image encoder
  on every frame (slower than SAM3VideoSemanticPredictor) and does NOT do
  temporal object tracking -- each frame is segmented independently.
- All three concepts are queried together in a single call per frame
  (`text=["road", "vehicle", "building"]`), and the single merged
  `Results.plot()` image from Ultralytics is used directly as the output
  frame (no manual mask compositing).
"""

import os
import cv2
from tqdm import tqdm

from ultralytics.models.sam import SAM3SemanticPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_VIDEO = "alpha.mp4"
OUTPUT_VIDEO = "alpha_masked.mp4"
TEXT_PROMPTS = ["road", "vehicle", "building"]

overrides = dict(
    conf=0.2,
    task="segment",
    mode="predict",
    model=os.path.join(os.environ["HOME"], "models", "sam3.pt"),
    half=True,
    quantize=16,
    compile=False,
    save=False,
    device="cuda",
)


def main():
    predictor = SAM3SemanticPredictor(overrides=overrides)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None  # created after first frame, once we know the plotted frame size

    frame_idx = 0
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing frames")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # SAM3SemanticPredictor.set_image() accepts a BGR np.ndarray
            # (same as cv2.imread output), so we can feed frames directly.
            predictor.set_image(frame)
            results = predictor(text=TEXT_PROMPTS)

            # results is a list of ultralytics.engine.results.Results;
            # for a single input image there is exactly one Results object.
            plotted = results[0].plot()

            if writer is None:
                out_h, out_w = plotted.shape[:2]
                writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (out_w, out_h))

            writer.write(plotted)

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Wrote {frame_idx} frames to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
