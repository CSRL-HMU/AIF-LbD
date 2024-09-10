import numpy as np
import sys
import select
import camera_class
import threading
from CSRL_orientation import *

# Initialize variables
filter_pole = 0.01
tracker = camera_class.HandTracker()
tracker.start_tracking()
# Get initial configuration

# # This is a function for getting asynchronously a key from the keyboard
# class NonBlockingConsole(object):
#
#     keyFlag = False
#
#     def __enter__(self):
#         self.old_settings = termios.tcgetattr(sys.stdin)
#         tty.setcbreak(sys.stdin.fileno())
#         return self
#
#     def __exit__(self, type, value, traceback):
#         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
#     def get_data(self):
#         if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
#             return sys.stdin.read(1)
#
#         return False


# this returns the finger position estimate from the camera
def get_pcf(firstTime, pcf_filtered):

    pcf_hat = np.array([0, 0, 0.3])

    isOcc = False

    # Get finger data

    fingertips3d_result, isOcc = tracker.get_fingertips3d()


    # print(fingertips3d_result)

    # if the array is not empty
    if fingertips3d_result and fingertips3d_result[0]["depth"]:

        # print(fingertips3d_result)
        ptemp = np.array([fingertips3d_result[0]["index_tip"].x, fingertips3d_result[0]["index_tip"].y,
                          fingertips3d_result[0]["depth"]])
        ptemp[0] = ptemp[0] * tracker.frame_width
        ptemp[1] = ptemp[1] * tracker.frame_height

        pcf_hat = tracker.ph.back_project(ptemp[0:2], ptemp[2])
        # if this the first time, initialize the state of the filter
        if firstTime:
            pcf_filtered = pcf_hat
            firstTime = False

        # State equation of the filter with integration
        pcf_filtered = filter_pole * pcf_hat + (1 - filter_pole) * pcf_filtered
    # print("test")
    # print(firstTime)
    # print(pcf_hat)
    # print(pcf_filtered)
    # print(isOcc)
    return firstTime, pcf_hat, pcf_filtered, isOcc


# this returns the robot's pose
def get_robot_pose(ur, q):

    # Get the hom. transform from robotics toolbox
    g = ur.fkine(q)
    # get rotation matrix
    R0e = np.array(g.R)
    # get translation
    p0e = np.array(g.t)

    # print('R0e=', R0e)
    # print('p0e=', p0e)

    # this is the pose of the camera with respec to the end-effector frame
    #Rec = rotZ(pi/2)
    Rec = np.identity(3)
    pec = np.array([-0.062, -0.025, 0.043 + 0.032])
    # pec = np.array([0.123, 0, 0])


    # compute the pose of the camera with respect to the inertial frame
    R0c = R0e @ Rec
    p0c = p0e + R0e @ pec

    return R0c, p0c


# this returns the robot's pose
def get_robot_UR_pose(rtde_r, ur):

    print("UR pose= ", rtde_r.getActualTCPPose())
    p = (rtde_r.getActualTCPPose())[0:3]

    # rx = 2.49
    # ry = -2.81
    # rz = -0.88
    # R = rxryrz_to_rotation((rtde_r.getActualTCPPose())[3], (rtde_r.getActualTCPPose())[4], (rtde_r.getActualTCPPose())[5])
    # R = angle2rot([(rtde_r.getActualTCPPose())[3], (rtde_r.getActualTCPPose())[4], (rtde_r.getActualTCPPose())[5]])

    # Get the hom. transform from robotics toolbox
    g = ur.fkine(rtde_r.getActualQ())

    # get rotation matrix
    R0e = np.array(g.R)
    # this is the pose of the camera with respec to the end-effector frame
    Rec = np.identity(3)
    # pec = np.array([0.123, 0, 0])
    # compute the pose of the camera with respect to the inertial frame
    R0c = R0e @ Rec

    print("UR p= ", rotZ(pi) @ p)
    print("UR R= ", R0c)

    return R0c, rotZ(pi) @ p


# returns the Jacobian of the robot (manera frame {C}) w.r.t. the world frame
def get_jacobian(ur, q):

    # get the Jacobian from robotics toolbox
    J = np.array(ur.jacob0(q))

    # get pose of the robot
    g = ur.fkine(q)
    p0e = np.array(g.t)
    R0e = np.array(g.R)

    pec = np.array([-0.062, -0.025, 0.043 + 0.032])
    p0c = p0e + R0e @ pec

    pce = p0e - p0c

    # COmpute the Jacobian for the cmaera frame
    GammaCE = np.identity(6)
    GammaCE[:3, -3:] = skewSymmetric(pce)
    J = GammaCE @ J

    return J



def calculate_dR_d(q, choice):
    if choice < 0 or choice > 3:
        print("Error in choice!")
        return
    q = np.array(q)
    q = q / np.linalg.norm(q)

    dr_d_ = np.zeros((3,3))

    if choice == 0:
        dr_d_[0,:] = [4 * q[0],    -2 * q[3],  2 * q[2]]
        dr_d_[1,:] = [2 * q[3],    4 * q[0],  -2 * q[1]]
        dr_d_[2,:] = [-2 * q[2],   2 * q[1],   4 * q[0]]

    elif choice == 1:
        dr_d_[0, :] = [4 * q[1],    2 * q[2],    2 * q[3]]
        dr_d_[1, :] = [2 * q[2],    0,          -2 * q[0]]
        dr_d_[2, :] = [2 * q[3],    2 * q[0],    0       ]

    elif choice == 2:
        dr_d_[0, :] = [0,           2 * q[1],   2 * q[0]]
        dr_d_[1, :] = [2 * q[1],    4 * q[2],   2 * q[3]]
        dr_d_[2, :] = [-2 * q[0],   2 * q[3],   0       ]
    else:
        dr_d_[0, :] = [0,           -2 * q[0],  2 * q[1]]
        dr_d_[1, :] = [2 * q[0],    0,          2 * q[2]]
        dr_d_[2, :] = [2 * q[1],    2 * q[2],   4 * q[3]]

    return dr_d_


