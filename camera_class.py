import numpy as np
import pyrealsense2 as rs
import cv2
import mediapipe as mp
import threading
import time
import pinhole

class HandTracker:
    def __init__(self):
        self.frame_width = 1280
        self.frame_height = 720
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.bgr8, 30)
        #self.config.enable_stream(rs.stream.depth, 840, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.depth, self.frame_width, self.frame_height, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.hole_filling = rs.hole_filling_filter(2)
        self.spatial_filter = rs.spatial_filter(0.5, 20, 2, 0)  # alpha, delta, magnitude, hole filling
        self.threshold_filter = rs.threshold_filter(0.2, 0.8)

        # Realsense 415
        # fx = 930.123
        # fy = 930.123
        # cx = 702.00
        # cy = 375.00

        # Realsense 435
        fx = 870.00
        fy = 900.00
        cx = 640.886
        cy = 363.087


        size_x = self.frame_width
        size_y = self.frame_height

        self.ph = pinhole.PinholeCamera(fx, fy, cx, cy, size_x, size_y)

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
            model_complexity=1
        )
        self.fingertips3d = []
        self.is_tracking = False
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.lock = threading.Lock()
        self.mp_drawing = mp.solutions.drawing_utils
        self.show_frame = False          # Toggle Show visualization window

    def _tracking_loop(self):
        while self.is_tracking:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not aligned_depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            frame_rgb = color_image
            #depth_image = np.asanyarray(aligned_depth_frame.get_data())
            results = self.hands.process(frame_rgb)


            try:
                if results.multi_hand_landmarks:
                    self.fingertips3d = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        if self.show_frame:
                            self.mp_drawing.draw_landmarks(frame_rgb, hand_landmarks)

                        fingertips = [
                            [int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x * self.frame_width),
                             int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y * self.frame_height),
                             hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].z],
                        ]

                        # print('ff= ', fingertips)

                        try:
                            depth_value = aligned_depth_frame.get_distance(fingertips[0][0], fingertips[0][1])
                            if depth_value < 0.1:
                                continue
                            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                            # print(f"intr:  {depth_intrin}")
                            # print(f"type:  {depth_intrin.type()}")
                            # point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [fingertips[0][0], fingertips[0][1]], depth_value)
                            point_3d = self.ph.back_project([fingertips[0][0], fingertips[0][1]], depth_value)
                            # point_3d[0] = point_3d[0]-0.014
                            # point_3d[2] = point_3d[2]  # + tip[2]
                            # print("point_3d= ", point_3d)

                            with self.lock:
                                self.fingertips3d.append(point_3d)
                        except:
                            continue

                        # print('depth_value= ', depth_value)


                else:
                    with self.lock:
                        self.fingertips3d = []

            except ValueError as ve:
                continue

            start_point = (640, 0)   # Coordinates of the starting point
            end_point = (640, 720)  # Coordinates of the ending point

            cv2.line(frame_rgb, (640, 0), (640, 720), (255, 255, 255), 1)
            cv2.line(frame_rgb, (0,360), (1280,360), (255, 255, 255), 1)

            if self.show_frame:
                cv2.imshow("hello", frame_rgb)
                cv2.waitKey(1)

    def start_tracking(self):
        if not self.is_tracking:
            self.is_tracking = True
            self.tracking_thread.start()

    def stop_tracking(self):
        if self.is_tracking:
            self.is_tracking = False
            self.tracking_thread.join()

    def get_fingertips3d(self):
        return self.fingertips3d


if __name__ == '__main__':

    # Example usage:
    tracker = HandTracker()
    tracker.start_tracking()

    # Add a delay or user input to simulate the program running for some time
    time.sleep(1)

    while True:
        start_time = time.time()
        with tracker.lock:
            fingertips3d_result = tracker.get_fingertips3d()
        loop_time = time.time() - start_time

        print("Fingertips 3D:", fingertips3d_result)
        #print(f'loop time: {loop_time}')

    tracker.stop_tracking()
    pipeline.stop()
    cv2.destroyAllWindows()
