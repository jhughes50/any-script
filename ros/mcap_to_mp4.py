#!/usr/bin/env python3
"""
Extract a sensor_msgs/msg/CompressedImage topic from a ROS 2 mcap bag and
write it out as an MP4 at a fixed frame rate.

Requires: source /opt/ros/jazzy/setup.bash  (for rosbag2_py, rclpy, sensor_msgs)
See bottom of this file / chat message for extra packages you need to install.

Usage:
    python3 mcap_to_mp4.py --bag /path/to/bag_dir_or_file.mcap \
        --topic /cam_driver/image_raw/compressed \
        --output out.mp4 \
        --fps 10
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


def get_reader(bag_path: str, storage_id: str = "mcap") -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True,
                         help="Path to bag directory (with metadata.yaml) or a raw .mcap file")
    parser.add_argument("--topic", default="/cam_driver/image_raw/compressed")
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video frame rate")
    parser.add_argument("--fourcc", default="mp4v",
                         help="OpenCV fourcc code, e.g. mp4v, avc1 (avc1 needs an ffmpeg-backed OpenCV build)")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"Bag path does not exist: {bag_path}", file=sys.stderr)
        sys.exit(1)

    reader = get_reader(str(bag_path))

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if args.topic not in type_map:
        print(f"Topic '{args.topic}' not found in bag. Available topics:", file=sys.stderr)
        for t in topic_types:
            print(f"  {t.name}  [{t.type}]", file=sys.stderr)
        sys.exit(1)

    if type_map[args.topic] != "sensor_msgs/msg/CompressedImage":
        print(f"Warning: topic type is {type_map[args.topic]}, "
              f"expected sensor_msgs/msg/CompressedImage", file=sys.stderr)

    storage_filter = rosbag2_py.StorageFilter(topics=[args.topic])
    reader.set_filter(storage_filter)

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = None
    frame_size = None
    out_period_ns = int(1e9 / args.fps)
    next_write_t = None
    read_count = 0
    written_count = 0

    while reader.has_next():
        topic, data, t = reader.read_next()  # t = recorded timestamp, nanoseconds

        msg = deserialize_message(data, CompressedImage)
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Warning: failed to decode frame at t={t}, skipping", file=sys.stderr)
            continue
        read_count += 1

        if writer is None:
            h, w = frame.shape[:2]
            frame_size = (w, h)
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, frame_size)
            if not writer.isOpened():
                print(f"Failed to open VideoWriter for '{args.output}' with fourcc "
                      f"'{args.fourcc}'. Try a different --fourcc (e.g. mp4v).", file=sys.stderr)
                sys.exit(1)
            next_write_t = t  # first output frame aligned to first message

        if frame.shape[1::-1] != frame_size:
            frame = cv2.resize(frame, frame_size)

        # Sample-and-hold resampling to a constant output fps:
        # write the most recent decoded frame for every output tick that has
        # elapsed. This upsamples (repeats frames) if input rate < fps, and
        # downsamples (drops frames) if input rate > fps.
        while next_write_t <= t:
            writer.write(frame)
            written_count += 1
            next_write_t += out_period_ns

    if writer is not None:
        writer.release()
    else:
        print("No frames were read from the topic; no video written.", file=sys.stderr)
        sys.exit(1)

    print(f"Read {read_count} frames from '{args.topic}', "
          f"wrote {written_count} frames to '{args.output}' at {args.fps} fps")


if __name__ == "__main__":
    main()
