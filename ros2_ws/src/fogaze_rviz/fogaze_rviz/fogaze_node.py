import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

_FOGAZE_ROOT = os.path.abspath(os.path.dirname(__file__))
for _ in range(10):
    if os.path.isfile(os.path.join(_FOGAZE_ROOT, "modules", "depth_estimator.py")):
        break
    _FOGAZE_ROOT = os.path.dirname(_FOGAZE_ROOT)
else:
    _FOGAZE_ROOT = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _FOGAZE_ROOT)

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, ColorRGBA
from builtin_interfaces.msg import Time

from modules.depth_estimator import DepthEstimator, OPTS
from modules.object_detector import ObjectDetector


FOCAL_LENGTH = 532.0
CX = 320.0
CY = 240.0


class FoGazeROSNode(Node):
    def __init__(self):
        super().__init__("fogaze_node")

        self.declare_parameter("yolo_model", "")
        self.declare_parameter("confidence", 0.5)
        self.declare_parameter("detection_interval", 3)
        self.declare_parameter("imgsz", 320)
        self.declare_parameter("camera_frame", "camera_color_optical_frame")

        yolo_model = self.get_parameter("yolo_model").value
        confidence = self.get_parameter("confidence").value
        det_interval = self.get_parameter("detection_interval").value
        imgsz = self.get_parameter("imgsz").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.pub_markers = self.create_publisher(MarkerArray, "detections/markers", 1)
        self.sub_pickup = self.create_subscription(
            String, "fogaze/pickup", self._on_pickup, 1)

        from pathlib import Path
        if not yolo_model:
            yolo_model = str(Path(_FOGAZE_ROOT) / "models" / "yolov8n.pt")
        self.get_logger().info(f"Loading YOLO ({yolo_model})...")
        self.detector = ObjectDetector(
            model_path=yolo_model, confidence=confidence, imgsz=imgsz)
        self.detector.set_detection_interval(det_interval)

        self.get_logger().info("Opening PrimeSense...")
        self.depth_est = DepthEstimator()
        for _ in range(30):
            self.depth_est.get_frame()

        self.get_logger().info("FoGaze ROS node ready")
        self.timer = self.create_timer(1.0 / 30.0, self._spin)

    def _on_pickup(self, msg):
        self.get_logger().info(f"PICKUP command: {msg.data}")

    def _spin(self):
        now = self.get_clock().now()
        stamp = now.to_msg()

        color_img, depth_img = self.depth_est.get_frame()
        if color_img is None:
            return

        h, w = color_img.shape[:2]
        detections = self.detector.detect(color_img)

        arr = MarkerArray()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            d = self.depth_est.depth_at_bbox(x1, y1, x2, y2)
            z_m = d / 100.0 if d is not None else 1.0
            cx_d = (x1 + x2) / 2.0
            cy_d = (y1 + y2) / 2.0
            x3d = (cx_d - CX) * z_m / FOCAL_LENGTH
            y3d = (cy_d - CY) * z_m / FOCAL_LENGTH

            box = Marker()
            box.header.stamp = stamp
            box.header.frame_id = self.camera_frame
            box.ns = "detections"
            box.id = i * 2
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = x3d
            box.pose.position.y = y3d
            box.pose.position.z = z_m
            w3d = abs(x2 - x1) * z_m / FOCAL_LENGTH
            h3d = abs(y2 - y1) * z_m / FOCAL_LENGTH
            box.scale.x = max(w3d, 0.02)
            box.scale.y = max(h3d, 0.02)
            box.scale.z = max(z_m * 0.1, 0.02)
            box.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.35)
            box.lifetime.sec = 1
            arr.markers.append(box)

            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = self.camera_frame
            label.ns = "labels"
            label.id = i * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x3d
            label.pose.position.y = y3d
            label.pose.position.z = z_m + 0.05
            label.scale.z = 0.06
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = f"{det['class_name']} {det['confidence']:.2f}"
            label.lifetime.sec = 1
            arr.markers.append(label)

        self.pub_markers.publish(arr)

    def close(self):
        self.depth_est.close()


def main(args=None):
    rclpy.init(args=args)
    node = FoGazeROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
