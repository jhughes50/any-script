#!/usr/bin/env python3
import rosbag
import numpy as np
import cv2
import argparse
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

parser = argparse.ArgumentParser(description="Process a bag file.")
parser.add_argument(
    "--bag",
    type=str,
    required=True,
    help="Path to the bag file"
)
parser.add_argument(
    "--topics",
    type=str,
    nargs="+",
    default=['/cam_driver/image_raw/compressed', '/vectornav/imu'],
    help="Topics to save to the new bag"
)
args = parser.parse_args()

bridge = CvBridge()
inbag = rosbag.Bag(args.bag)

output_name = args.bag.split(".")[0] + "_uc.bag"

with rosbag.Bag(output_name, 'w') as outbag:
    for topic, msg, t in inbag.read_messages(raw=True):
        if topic not in args.topics:
            continue

        # Raw msg: (datatype, serialized_data, md5, position, pytype)
        msg_type = msg[0]

        if msg_type == 'sensor_msgs/CompressedImage':
            raw_data = msg[1]

            # Extract the JPEG/PNG data from the raw serialized bytes
            # CompressedImage layout: header + format_string + data_bytes
            # We can find the image data by looking for JPEG/PNG magic bytes
            jpeg_start = raw_data.find(b'\xff\xd8')  # JPEG magic
            png_start = raw_data.find(b'\x89PNG')     # PNG magic

            if jpeg_start >= 0:
                img_bytes = raw_data[jpeg_start:]
            elif png_start >= 0:
                img_bytes = raw_data[png_start:]
            else:
                print(f"Skipping message at {t} - no recognizable image data")
                continue

            np_arr = np.frombuffer(img_bytes, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                print(f"Skipping message at {t} - failed to decode")
                continue

            img_msg = bridge.cv2_to_imgmsg(cv_img, "bgr8")
            img_msg.header.stamp = t
            out_topic = topic[:-len("/compressed")] if topic.endswith("/compressed") else topic
            outbag.write(out_topic, img_msg, t)
        else:
            # Write other topics normally (they may also fail if mismatched)
            try:
                topic_name, data, md5, pos, pytype = msg
                outbag.write(topic, pytype().deserialize(data), t)
            except Exception:
                pass  # skip topics that also can't deserialize

inbag.close()
