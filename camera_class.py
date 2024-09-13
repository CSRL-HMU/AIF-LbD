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
        runtime_params = sl.RuntimeParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = 0.2
        self.zed.open(init_params)

        self.runtime_params = sl.RuntimeParameters()
        self.mat = sl.Mat()
        self.depth = sl.Mat()

        # ZED camera intrinsic parameters
        fx = 533.895
        fy = 534.07
        cx = 632.69
        cy = 379.6325

        self.isOccluded = False

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
        #self.target_class = self.yolo.names  # Specify the target class
        self.target_class =['bicycle','car','motorcycle','airplane','bus','train','truck', 'boat','traffic light','fire hydrant', 'stop sign','parking meter','bench','bird', 'cat','dog','horse','sheep','cow','elephant','bear', 'zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball', 'kite','baseball bat','baseball glove', 'skateboard', 'surfboard','tennis racket', 'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair', 'couch','potted plant', 'bed', 'dining table','toilet','tv','laptop','mouse','remote','keyboard', 'cell phone', 'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']


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
            return self.fingertips3d, self.isOccluded

    def detect_objects(self, frame):
        # Convert frame to BGR format (3 channels) if necessary
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        results = self.yolo(frame,verbose=False)
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

                #print(f"Detected class: {class_name}, Target class: {self.target_class}")  # Debugging information

                #if class_name != self.target_class:
                if class_name not in self.target_class:
                    continue

                object_depth = self.get_depth((xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2)

                if object_depth is None:
                    continue

                # print(f"Index tip: ({index_x}, {index_y}, {index_depth}), Box: ({xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}, {object_depth}), Class: {class_name}")

                if xyxy[0] <= index_x <= xyxy[2] and xyxy[1] <= index_y <= xyxy[3] and (index_depth > object_depth or math.isinf(index_depth)):
                    # print(f"Index finger is occluded by a {class_name}!")
                    return True

                # Additional debugging to understand why the condition might fail
                # print(f"Condition check details: {xyxy[0] <= index_x <= xyxy[2]}, {xyxy[1] <= index_y <= xyxy[3]}, {index_depth >object_depth}")
                # print(f"Bounding Box: ({xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}), Index Finger: ({index_x}, {index_y}), Depths: Index {index_depth}, Object {object_depth}")

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
            # print('test1')
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # print('test2')
                for landmark in hand_landmarks.landmark:
                    # print('test3')
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    # print('test4')
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    # print('test5')

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
                    # print('test6')
                    with self.lock:
                    # print('test7')
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
                    self.isOccluded = False
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        if self.is_index_finger_occluded(index_tip, self.detections):
                            # print("!!!!--OCCLUDED!!!!!!!!!")
                            self.isOccluded = True

                cv2.imshow("Frame", frame)
                # cv2.imwrite('./image1.png', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":


    # [LEFT_CAM_HD]
    # fx=533.895
    # fy=534.07
    # cx=632.69
    # cy=379.6325
    # k1=-0.0489421
    # k2=0.0208547
    # p1=0.000261529
    # p2=-0.000580449
    # k3=-0.00836067
    #
    # [RIGHT_CAM_HD]
    # fx=532.225
    # fy=532.47
    # cx=645.515
    # cy=362.0185
    # k1=-0.0463267
    # k2=0.0195163
    # p1=0.000313832
    # p2=-8.13248e-05
    # k3=-0.00854262

    # Example usage:
    tracker = HandTracker()
    tracker.start_tracking()

    # Add a delay or user input to simulate the program running for some time
    time.sleep(1)

    isOcc = False

    while True:
        # print('ttttest1')
        # swith tracker.lock:tart_time = time.time()

            # print('Test!')
        fingertips3d_result, isOcc = tracker.get_fingertips3d()
        if fingertips3d_result:
            p = np.array([fingertips3d_result[0]["index_tip"].x, fingertips3d_result[0]["index_tip"].y, fingertips3d_result[0]["depth"]])
            p[0] = p[0] * tracker.frame_width
            p[1] = p[1] * tracker.frame_height

            pp = tracker.ph.back_project(p[0:2],p[2])
            print("--------------------")
            print(pp)
            print(isOcc)

        # print('ttttest2')

        # loop_time = time.time() - start_time

        # print("Fingertips 3D:", fingertips3d_result)
        #print(f'loop time: {loop_time}')
        time.sleep(1)

    tracker.stop_tracking()
    tracker.zed.close()
    cv2.destroyAllWindows()

####

