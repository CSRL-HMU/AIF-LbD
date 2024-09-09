import msvcrt
import roboticstoolbox as rt
import numpy as np
import scipy.io as scio
import spatialmath as sm
import rtde_receive
import rtde_control
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import time
from CSRL_orientation import *
from AIF_aux import *
import threading
from scipy.spatial.distance import euclidean
from dmpR3 import *
import camera_class

# Declare math pi
pi = math.pi

# ip_robot = "192.168.1.60"     # for UR3e
ip_robot = "192.168.1.100"      # for UR5e

# Define robot
rtde_c = rtde_control.RTDEControlInterface(ip_robot)
rtde_r = rtde_receive.RTDEReceiveInterface(ip_robot)


# # commanded initial configuration for experimenting
# up UR3e
# q0d = np.array([-2.7242565790759485, -2.014890810052389, -1.7824010848999023, -0.8763185304454346, 1.5800056457519531, -1.227436367665426])

# side UR3e
# q0d = np.array([-3.6925345102893274, -2.9910251102843226, -1.1009764671325684, -1.0728790921023865, 2.3592722415924072, -2.264323059712545])

# side UR5e
# q0d = np.array([-0.41690332094301397, -1.883740564385885, -1.7535805702209473, -1.4085637044957657, 2.464108467102051, 1.0600205659866333])
# q0d = np.array([0.3244214355945587, -1.9122115574278773, -1.2255409955978394, -1.4460056287101288, 1.4586446285247803, 1.7281198501586914])
# q0d = np.array([1.632409930229187, 0.016547008151672316, 1.609415356312887, -3.1515056095519007, 0.759623110294342, 4.654907703399658])

# Ellipse exp
# q0d = np.array([-0.5215924421893519, -2.0054155788817347, -1.5068203210830688, -1.4493384447744866, 2.4010772705078125, 1.0605438947677612 - 1.57])
# q0d = np.array([-0.3801339308368128, -1.9773642025389613, -1.316433310508728, -1.7317115269102992, 2.124098539352417, -0.4773214499102991])

# Fruit exp
q0d = np.array([1.182332992553711, -2.2384849987425746, -1.71377694606781, -3.0127340755858363, 0.5584464073181152, 2.2159714698791504])
# q0d = np.array([0.7912634611129761, -1.980398794213766, -1.1590644121170044, -1.9489723644652308, 0.9563465714454651, 2.411074638366699 - 1.57])


# Move to the initial configuration
rtde_c.moveJ(q0d, 0.5, 0.5)

# Get initial configuration
q0 = np.array(rtde_r.getActualQ())

# Create the robot
# ur = rt.models.UR3()
# ur = rt.models.UR5()

# define the robot with its DH parameters
ur = rt.DHRobot([
    rt.RevoluteDH(d = 0.1625, alpha = pi/2),
    rt.RevoluteDH(a = -0.425),
    rt.RevoluteDH(a = -0.3922),
    rt.RevoluteDH(d = 0.1333, alpha = pi/2),
    rt.RevoluteDH(d = 0.0997, alpha = -pi/2),
    rt.RevoluteDH(d = 0.0996)
],name='UR5e')

# Get the initial robot state
Rp, pp = get_robot_pose(ur, q0)

# define the center of the scene
pc = pp + Rp @ 0.4

# Control cycle
dt = 0.002

# Init time
t = 0.0

# get time now
t_now = time.time()

# blocking
data_id = input('Press any key to start motion recording. This will be stored as the data ID: ')

# Uncomment if you want to free drive[-0.7225549856769007, -1.5372941692224522, -2.1203153133392334, -0.1600521367839356, 1.5096690654754639, 3.262672185897827]
# rtde_c.freedriveMode()

# Control gains
ka = 1.0

# This is a flag for indicating the first iteration
firstTime = True

# Initialization of p_hat ... this passes through !!!!!!!! hstack !!!!!
p_hat = pp + Rp*np.array([[0], [0], [0.3]])

# initialize filtered estimation
pcf_filtered = np.array([[0], [0], [0.3]])


# Initialize logging
plog = p_hat
tlog = np.array([t])

# Set the covariance matrix of the camera
sigma_1 = 0.001
sigma_2 = 0.05
Sigma_0 = np.identity(3)
Sigma_0[0, 0] = sigma_1
Sigma_0[1, 1] = sigma_1
Sigma_0[2, 2] = sigma_2

# Total discrete time is initialy set to a very large number (we don't know this before the first iteration)
Nk = 100000  # T = Nk * dt (for Nk=100000 -> T=200s)

# Number of DMP kernels
M = 40


# this is the initial pf 
pp.shape = (3, 1)
p_hat.shape = (3, 1)



# Initialization of P
P = np.zeros((3,3*Nk))
for k in range(Nk):
    P[0:3, k*3:k*3+3] = 100.0 * np.identity(3)

# Initialization of Q
Q = P
Sigma_log = P
    

# FOR EACH ITERATION (3 max are considered)
for i in range(2):

    # get the robot's state
    Rp, pp = get_robot_pose(ur, q0)
    pp.shape = (3, 1)

    print('Initiating iteration No: ', i)

    # Initialize the arrays that pass through !!!!!!!! hstack !!!!!
    p_hat_iter = np.array([[0], [0], [0.3]])
    qp_iter = np.array([[1], [0], [0], [0]])  # QUATERNION
    pp_iter = pp
    pp_iter.shape = (3, 1)

    # dummy initialization of the DMP initial position and target
    pT = p_hat
    x1 = p_hat

    if i == 1:
        # go to a new random configuration
        # for UR3e
        # q0d = np.array(
        #     [-2.7242565790759485, -2.014890810052389, -1.7824010848999023, -0.8763185304454346, 1.5800056457519531,
        #      -1.227436367665426])

        # for UR5e
        # exp ellipse
        # q0d = np.array([2.0775132179260254, -1.5693469533328717, 1.0836318174945276, -1.3998232048800965, -1.333468262349264, 5.02064323425293])
        # q0d = np.array(
        #     [0.7912634611129761, -1.980398794213766, -1.1590644121170044, -1.9489723644652308, 0.9563465714454651, 2.411074638366699 - 1.57])

        # exp fruit
        q0d = np.array([1.2330601215362549, -1.2422520977309723, -1.6271378993988037, -2.3346992931761683, 0.3470206558704376,
             0.5055640339851379])

        Q_now = Q[0:3, 0:3]
    # END OF IF


    # Move to the initial configuration
    rtde_c.moveJ(q0d, 0.5, 0.5)

    # Wait for a finger to appear
    while True:
        with tracker.lock:
            fingertips3d_result = tracker.get_fingertips3d()
        # END OF WITH

        time.sleep(0.002)
        # print(fingertips3d_result)
        if fingertips3d_result:
            break
        # END OF IF
    # END OF WHILE

    p_hat.shape = (3, 1)

    # after a finger is detected, wait for 3 seconds
    time.sleep(3)

    # make a boop sound for the humnan to start the demonstration
    beep = lambda x: os.system("echo -n '\a';sleep 0.015;" * x)
    beep(10)



    # This variable is only for the anti-spike filter
    # p_prev = p_hat

    # log time
    tlog = np.array([t])


    # DMP initialization, with many default selections, e.g. Gaussian kernel, Linear Canonical system, alpha, beta ... 
    # M is the number of Kernels
    dmp_model = dmpR3(M, 20)
    dmp_model.set_goal(pT)


    # initialize time and discrete time
    t = 0
    k = 0

    # this is for accepting commands from keyboard
    # with NonBlockingConsole() as nbc:

    # while the reference index is less that the size of the reference time array
    while k < Nk:     # CONTROL LOOP

        # Start control loop - synchronization with the UR
        t_start = rtde_c.initPeriod()

        # Integrate time
        t = t + dt

        # Get joint values
        q = np.array(rtde_r.getActualQ())

        # get state
        Rp, pp = get_robot_pose(ur, q)

        # initialize v_p
        v_p = np.zeros(6)


        # get the current estimation of the finger
        firstTime, pcf_hat, pcf_filtered = get_pcf(firstTime, pcf_filtered)
        p_hat = pp + Rp @ pcf_filtered

        # initialize DMP state
        if k == 0:
            x1 = p_hat
            dmp_model.set_init_state(x1)


        # get the Jacobian
        J = get_jacobian(ur, q)

        # shape arrays before stacking them
        p_hat.shape = (3, 1)
        pp.shape = (3, 1)
        Rp.shape = (3, 3)
        qp = rot2quat(Rp)
        qp.shape = (4, 1)


        # log time
        tlog = np.vstack((tlog, t))

        # compute Sigma now
        Sigma_now = Rp @ Sigma_0 @ Rp.T
        Sigma_inv = np.linalg.inv(Sigma_now)

        # log Sigma
        Sigma_log[0:3, k*3:k*3+3] = Sigma_now

        # get modelling uncertainty,i.e. Q
        Q_now = Q[0:3, k*3:k*3+3]
        Q_inv = np.linalg.inv(Q_now)

        # State dot prediction though DMP
        z_dot, p_dot, p_2dot = dmp_model.get_state_dot(x3 , x1, x2)

        # state integration
        x1 = x1 + p_dot * dt
        x2 = x2 + p_2dot * dt
        x3 = x3 + z_dot * dt

        # Weighted Mean Data Fusion and its covariance
        P[0:3, k*3:k*3+3] = np.linalg.inv( Q_inv + Sigma_inv )
        pstar = P[0:3, k*3:k*3+3] @ ( Q_inv @ x1 + Sigma_inv @ p_hat )

        # stack data array
        p_hat_iter = np.hstack((p_hat_iter, np.array(p_hat)))
        qp_iter = np.hstack((qp_iter, qp))
        pp_iter = np.hstack((pp_iter, np.array(pp)))

        ## ACtive perception signal
        detP = np.linalg.det(P[0:3, k*3:k*3+3])
        invP = np.linalg.inv(P[0:3, k*3:k*3+3])
        Jq = getJq(qp)
        Spp = skewSymmetric(pc-pp)

        A = P[0:3, k*3:k*3+3] @ P[0:3, k*3:k*3+3] @ Sigma_inv @ Sigma_inv

        ddet_dq = np.zeros(4)
        for j in range (4):
            dR_dqi = calculate_dR_d(qp, j)
            dSigma_dqi = dR_dqi @ Sigma_now @ np.transpose(Rp) + Rp @  Sigma_now @ np.transpose(dR_dqi)
            dP_dqi = A @ dSigma_dqi
            ddet_dq[j] = detP @ np.trace( invP @ dP_dqi )

        v_p[0:3] = - ka * Spp @ np.transpose(Jq) @ ddet_dq
        v_p[3:6] = - ka * np.transpose(Jq) @ ddet_dq

        # Inverse kinematics mapping with siongularity avoidance
        qdot = np.linalg.pinv(J, 0.1) @ ( v_p )

        # if the key a is pushed
        if msvcrt.kbhit():
            if msvcrt.getch() == 'a':
                Nk = k
                print('Stopping the iteration')
                break
            # END OF IF
        # END OF IF

        # set joint speed with acceleration limits
        rtde_c.speedJ(qdot, 1.0, dt)

        # This is for synchronizing with the UR robot
        rtde_c.waitPeriod(t_start)

    # END OF WHILE -- control loop

    # Stop velocity control 
    rtde_c.speedStop()

    # make a sound (beep)
    beep = lambda x: os.system("echo -n '\a';sleep 0.015;" * x)
    beep(5)
   
    # Re-set the target
    pT = p_hat

    # Train the DMP model
    dmp_model.set_goal(pT)
    dmp_model.train(dt, p_hat_iter, plotPerformance = True)

    # write the log files
    data = {'p_hat_iter': p_hat_iter, 'qp_iter': qp_iter, 'pp_iter': pp_iter, 't': tlog, 'P': P, 'Q': Q, 'Sigma_log':Sigma_log}
    scio.savemat('Logging_' + str(data_id) + '_' + str(i) + '.mat', data)


# END OF (BIG) FOR .... ITERATIONS

# stop the robot
rtde_c.speedStop()
rtde_c.stopScript()










