import os
import argparse
import cv2
import torch
from tqdm import tqdm
from ultralytics.models.sam import SAM3SemanticPredictor

TEXT_PROMPTS = ["road", "military vehicle", "building", "person"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 semantic segmentation over a video.")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video file")
    parser.add_argument("--model", required=True, help="Path to SAM3 model weights")
    return parser.parse_args()


def _filter(result):
    mask_tensor = result.masks.data
    masks_bool = mask_tensor > 0.5
    confidences = result.boxes.conf

    sorted_indices = torch.argsort(confidences, descending=True)

    keep_indices = []

    for i in sorted_indices:
        current_mask = masks_bool[i]
        current_conf = confidences[i].item()
        is_duplicate = False

        for kept_idx in keep_indices:
            winner_mask = masks_bool[kept_idx]

            intersection = (current_mask & winner_mask).sum().float()
            current_area = current_mask.sum().float()

            if current_area > 0:
                overlap_ratio = intersection / current_area
            else:
                overlap_ratio = 0.0

            if overlap_ratio > 0.9:
                is_duplicate = True
                break

        if not is_duplicate:
            keep_indices.append(i.item())
    return result[keep_indices]


def main():
    args = parse_args()

    input_video = args.input
    output_video = args.output
    model_path = args.model

    overrides = dict(
        conf=0.3,
        task="segment",
        mode="predict",
        model=model_path,
        half=False,
        compile=False,
        save=False,
        device="cuda",
    )

    predictor = SAM3SemanticPredictor(overrides=overrides)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {input_video}")

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
            results = predictor(text=TEXT_PROMPTS)[0]
            if results.masks:
                results = _filter(results)
            # results is a list of ultralytics.engine.results.Results;
            # for a single input image there is exactly one Results object.
            plotted = results.plot()
            if writer is None:
                out_h, out_w = plotted.shape[:2]
                writer = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))
            writer.write(plotted)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Wrote {frame_idx} frames to {output_video}")


if __name__ == "__main__":
    main()
