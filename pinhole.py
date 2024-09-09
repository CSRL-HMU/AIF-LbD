import numpy as np

class PinholeCamera:
    def __init__(self, fx, fy, cx, cy, size_x, size_y):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.size_x = size_x
        self.size_y = size_y

    def project(self, point_3d):
        point = np.zeros(3)
        point[0] = point_3d[0] * self.fx / point_3d[2] + self.cx
        point[1] = point_3d[1] * self.fy / point_3d[2] + self.cy
        point[2] = point_3d[2]
        return point

    def back_project(self, point, depth_value):
        point_3d = np.zeros(3)
        point_3d[2] = depth_value
        point_3d[0] = (point[0] - self.cx) * point_3d[2] / self.fx
        point_3d[1] = (point[1] - self.cy) * point_3d[2] / self.fy
        return point_3d

    def matrix(self):
        cam_matrix = np.zeros((3, 3))
        cam_matrix[0, 0] = self.fx
        cam_matrix[1, 1] = self.fy
        cam_matrix[0, 2] = self.cx
        cam_matrix[1, 2] = self.cy
        cam_matrix[2, 2] = 1.0
        return cam_matrix
