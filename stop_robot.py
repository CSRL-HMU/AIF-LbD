
import roboticstoolbox as rt
import numpy as np
import rtde_receive
import rtde_control
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from CSRL_orientation import *

# Declare math pi
pi = math.pi

# ip_robot = "192.168.1.60"     # for UR3e
ip_robot = "192.168.1.100"      # for UR5e

# Define robot
rtde_c = rtde_control.RTDEControlInterface(ip_robot)
rtde_r = rtde_receive.RTDEReceiveInterface(ip_robot)

# stop the robot
rtde_c.speedStop()
rtde_c.stopScript()










