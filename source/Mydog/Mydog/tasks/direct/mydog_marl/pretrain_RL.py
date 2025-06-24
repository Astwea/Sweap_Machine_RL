import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import cv2
import math
import threading

class LaserCameraOverlay(Node):
    def __init__(self):
        super().__init__('laser_camera_overlay')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.lidar_data = None

        # === 相机内参 ===（请根据实际相机修改）
        fx, fy = 438.78, 437.30
        cx, cy = 305.59, 243.74
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])

        # === 雷达到相机的平移 ===
        self.t = np.array([[-0.155], [0.0], [0.02]])  # 单位：米（你提供的）
        self.R = self.get_R_lidar_to_cam_with_offset(-45)


        # 摄像头初始化
        self.cap = cv2.VideoCapture(0)
        self.laser_overlay = np.zeros((480, 640, 3), dtype=np.uint8)  # 注意分辨率
        self.laser_lock = threading.Lock()
    def scan_callback(self, msg):
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = np.array(msg.ranges)

        angle_array = angle_min + np.arange(len(ranges)) * angle_increment
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)

        overlay = np.zeros((480, 640, 3), dtype=np.uint8)

        for r, theta in zip(ranges[valid_mask], angle_array[valid_mask]):
            result = self.lidar_to_pixel(r, theta)
            if result is None:
                continue
            (u, v), dist = result
            if 0 <= u < 640 and 0 <= v < 480:
                color = self.get_colormap_color(dist)
                cv2.circle(overlay, (u, v), 2, color, -1)

        with self.laser_lock:
            self.laser_overlay = overlay


    def get_R_lidar_to_cam_with_offset(self, angle_offset_deg):
        angle = np.deg2rad(angle_offset_deg)
        R_base = np.array([
            [0, -1,  0],
            [0,  0, -1],
            [1,  0,  0]
        ])
        R_z = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
        return R_base @ R_z

    def lidar_to_pixel(self, r, theta):
        x_l = r * math.cos(theta)
        y_l = r * math.sin(theta)
        z_l = 0
        p_lidar = np.array([[x_l], [y_l], [z_l]])  # shape (3,1)

        p_cam = self.R @ p_lidar + self.t
        X, Y, Z = p_cam[0, 0], p_cam[1, 0], p_cam[2, 0]
        #if theta == 3.14:
            #print(f"雷达点 {r:.2f}m @ {theta:.2f}rad -> 相机坐标: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")

        if Z <= 0:
            return None
        p_img = self.K @ (p_cam / Z)
        u, v = int(p_img[0, 0]), int(p_img[1, 0])
        return (u, v), r

    def get_colormap_color(self, r, r_min=0.3, r_max=3.0):
        r = max(r_min, min(r, r_max))
        value = int(255 * (1.0 - (r - r_min) / (r_max - r_min)))
        color_map = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)
        return tuple(int(c) for c in color_map[0, 0])  # (B, G, R)

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.laser_lock:
                overlay_copy = self.laser_overlay.copy()
            print(frame.shape)
            #blended = cv2.addWeighted(frame, 1.0, overlay_copy, 1.0, 0)
            cv2.imshow("Laser Projection", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = LaserCameraOverlay()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
