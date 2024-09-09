import numpy as np
import cv2
import mediapipe as mp
import threading
import time
import pyzed.sl as sl
import pinhole
from ultralytics import YOLO
import supervision as sv
import logging
import math
class HandTracker:
    def __init__(self):
        self.frame_width = 1280
        self.frame_height = 720

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        self.zed.open(init_params)

        self.runtime_params = sl.RuntimeParameters()
        self.mat = sl.Mat()
        self.depth = sl.Mat()

        # ZED camera intrinsic parameters 
        fx = 700.819
        fy = 700.819
        cx = 665.465
        cy = 371.953

        size_x = self.frame_width
        size_y = self.frame_height

        self.ph = pinhole.PinholeCamera(fx, fy, cx, cy, size_x, size_y)

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.yolo = YOLO("yolov8n.pt")  # Load YOLO model
        self.detections = []
        self.target_class = "apple"  # Specify the target class

        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.lock = threading.Lock()
        self.is_tracking = False
        self.fingertips3d = []

    def start_tracking(self):
        if not self.is_tracking:
            self.is_tracking = True
            self.tracking_thread.start()

    def stop_tracking(self):
        if self.is_tracking:
            self.is_tracking = False
            self.tracking_thread.join()


    def get_fingertips3d(self):
        with self.lock:
            return self.fingertips3d

    def detect_objects(self, frame):
        # Convert frame to BGR format (3 channels) if necessary
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        results = self.yolo(frame)
        self.detections = results  # Get detections

    def detect_hands(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results

    def get_depth(self, x, y):
        err = self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        if err == sl.ERROR_CODE.SUCCESS:
            depth_value = self.depth.get_value(int(x), int(y))
            if not np.isnan(depth_value[1]):
                return depth_value[1]  # depth_value is a tuple (err, depth) where err is the error code for the depth retrieval at (x, y)
        return None

    def is_index_finger_occluded(self, index_tip, detections):
        index_x = int(index_tip.x * self.frame_width)
        index_y = int(index_tip.y * self.frame_height)
        index_depth = self.get_depth(index_x, index_y)

        if index_depth is None:
            return False

        for detection in detections:
            for box in detection.boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                class_id = int(box.cls[0].cpu().numpy())  # Class ID of the detected object
                class_name = detection.names[class_id]  # Get class name from YOLO model

                print(f"Detected class: {class_name}, Target class: {self.target_class}")  # Debugging information

                if class_name != self.target_class:
                    continue

                object_depth = self.get_depth((xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2)

                if object_depth is None:
                    continue

                print(f"Index tip: ({index_x}, {index_y}, {index_depth}), Box: ({xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}, {object_depth}), Class: {class_name}")

                if xyxy[0] <= index_x <= xyxy[2] and xyxy[1] <= index_y <= xyxy[3] and (index_depth > object_depth or math.isinf(index_depth)):
                    print(f"Index finger is occluded by a {class_name}!")
                    return True

                # Additional debugging to understand why the condition might fail
                print(f"Condition check details: {xyxy[0] <= index_x <= xyxy[2]}, {xyxy[1] <= index_y <= xyxy[3]}, {index_depth >object_depth}")
                print(f"Bounding Box: ({xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}), Index Finger: ({index_x}, {index_y}), Depths: Index {index_depth}, Object {object_depth}")

        return False

    def draw_detections(self, frame):
        # Draw YOLO detections
        for detection in self.detections:
            for box in detection.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())  # Class ID of the detected object
                class_name = detection.names[class_id]  # Get class name from YOLO model
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    def draw_hands(self, frame, hand_results):
        # Draw MediaPipe hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    def _tracking_loop(self):
        while self.is_tracking:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.mat, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
                frame = self.mat.get_data()

                # Perform object detection
                self.detect_objects(frame)

                # Perform hand detection
                hand_results = self.detect_hands(frame)

                # Draw detections on frame
                self.draw_detections(frame)
                self.draw_hands(frame, hand_results)

                if hand_results.multi_hand_landmarks:
                    with self.lock:
                        self.fingertips3d = [
                            {
                                "index_tip": hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP],
                                "depth": self.get_depth(
                                    int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * self.frame_width),
                                    int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * self.frame_height)
                                )
                            }
                            for hand_landmarks in hand_results.multi_hand_landmarks
                        ]
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        if self.is_index_finger_occluded(index_tip, self.detections):
                            print("!!!!--OCCLUDED!!!!!!!!!")

                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
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
        time.sleep(1)

    tracker.stop_tracking()
    tracker.zed.close()
    cv2.destroyAllWindows()

####

