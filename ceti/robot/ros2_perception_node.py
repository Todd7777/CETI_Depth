#!/usr/bin/env python3
"""
ROS2 perception node for CETI lab robots.

Subscribes to camera images, publishes depth maps and whale detections.
Requires: rclpy, sensor_msgs, cv_bridge (install via apt/rosdep).

Usage:
    source /opt/ros/humble/setup.bash
    python ceti/robot/ros2_perception_node.py --encoder vits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="CETI ROS2 perception node")
    parser.add_argument("--encoder", type=str, default=None, choices=["vits", "vitb", "vitl"])
    parser.add_argument("--depth-checkpoint", type=str, default=None)
    parser.add_argument("--image-topic", type=str, default="/robot/camera/image_raw")
    parser.add_argument("--depth-topic", type=str, default="/ceti/depth")
    parser.add_argument("--detection-topic", type=str, default="/ceti/whale_detections")
    parser.add_argument("--whale-checkpoint", type=str, default=None)
    parser.add_argument("--underwater-preprocess", action="store_true")
    parser.add_argument("--rate-hz", type=float, default=10.0)
    args = parser.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        from cv_bridge import CvBridge
    except ImportError:
        print("ROS2 dependencies not installed.")
        print("Install with:")
        print("  sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs")
        print("  pip install rclpy")
        sys.exit(1)

    import torch
    from ceti.depth.infer_robot import (
        build_depth_model,
        build_whale_detector,
        process_frame,
        resolve_encoder,
    )

    from ceti.utils.device import get_device

    device = str(get_device())
    depth_enc = resolve_encoder(args.depth_checkpoint, args.encoder)
    depth_model, transform = build_depth_model(depth_enc, device, args.depth_checkpoint)
    whale_detector = build_whale_detector(args.whale_checkpoint)

    class CETIPerceptionNode(Node):
        def __init__(self):
            super().__init__("ceti_perception")
            self.bridge = CvBridge()
            self.sub = self.create_subscription(
                Image, args.image_topic, self.image_callback, 10
            )
            self.depth_pub = self.create_publisher(Image, args.depth_topic, 10)
            self.det_pub = self.create_publisher(String, args.detection_topic, 10)
            self.get_logger().info(
                f"CETI perception active: {args.image_topic} → {args.depth_topic}"
            )

        def image_callback(self, msg: Image):
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"CV bridge error: {e}")
                return

            vis, detections = process_frame(
                frame, depth_model, transform, whale_detector, device,
                args.underwater_preprocess, "combined", 0.5,
            )

            # Publish depth visualization
            depth_msg = self.bridge.cv2_to_imgmsg(vis[:, frame.shape[1]:], "bgr8")
            depth_msg.header = msg.header
            self.depth_pub.publish(depth_msg)

            # Publish detections as JSON string
            if detections:
                import json
                det_msg = String()
                det_msg.data = json.dumps(detections)
                self.det_pub.publish(det_msg)

    rclpy.init()
    node = CETIPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
