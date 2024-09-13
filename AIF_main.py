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
import winsound
import keyboard

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
# q0d = np.array([1.182332992553711, -2.2384849987425746, -1.71377694606781, -3.0127340755858363, 0.5584464073181152, 2.2159714698791504])
# q0d = np.array([0.7912634611129761, -1.980398794213766, -1.1590644121170044, -1.9489723644652308, 0.9563465714454651, 2.411074638366699 - 1.57])

# AIF exp
# Initial
# q0d = [-0.30383807817568, -1.47846573710952, -1.9760072231292725, -1.7141696415343226, 2.122706651687622, -0.9522011915790003]

#control experiment
q0d = [0.02048109658062458, -1.4097058337977906, -2.217151641845703, -1.0382676881602784, 2.213860034942627, -1.263026539479391]
# Move to the initial configuration
rtde_c.moveJ(q0d, 0.5, 0.5)

fx = tracker.ph.fx
fy = tracker.ph.fy

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
# pc = pp + Rp @ np.array([0, 0, 0.4])
pc = np.array([0.27, 0.15, 0])
pc.shape = (3, 1)

# Control cycle
dt = 0.002

# Init time
t = 0.0

# get time now
t_now = time.time()

# blocking
data_id = input('Press any key to start motion recording. This will be stored as the data ID: ')

# Uncomment if you want to free drive
# rtde_c.freedriveMode()

# Control gains
ka = 30000.0

# This is a flag for indicating the first iteration
firstTime = True

# Initialization of p_hat ... this passes through !!!!!!!! hstack !!!!!
p_hat = pp + Rp @ np.array([0, 0, 0.3])

# initialize filtered estimation
pcf_filtered = np.array([0, 0, 0.3])


# Initialize logging
plog = p_hat
tlog = np.array([t])

# Set the covariance matrix of the camera
sigma_1 = 40.0
sigma_2 = 40.0
sigma_3 = 0.3


depth_calibration_param = -0.005


# Total discrete time is initialy set to a very large number (we don't know this before the first iteration)
Nk = 100000  # T = Nk * dt (for Nk=100000 -> T=200s)

# Number of DMP kernels
M = 100


# this is the initial pf 
pp.shape = (3, 1)
p_hat.shape = (3, 1)


# Initialization of P
P = np.zeros((3, 3*Nk))
for k in range(Nk):
    P[0:3, k*3:k*3+3] = 10000.0 * np.identity(3)

# Initialization of Q
Q = np.array(P)
Sigma_log = np.array(P)

dmp_model = dmpR3(M, 20)

# FOR EACH ITERATION (3 max are considered)
for i in range(3):

    # get the robot's state
    Rp, pp = get_robot_pose(ur, q0)
    pp.shape = (3, 1)

    print('Initiating iteration No: ', i)

    # Initialize the arrays that pass through !!!!!!!! hstack !!!!!
    p_hat_iter = np.array([[0], [0], [0.3]])
    qp_iter = np.array([[1], [0], [0], [0]])  # QUATERNION
    pp_iter = pp
    pp_iter.shape = (3, 1)
    pstar_log = np.array([[0], [0], [0]])
    x1_log = np.array([[0], [0], [0]])



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
        # q0d = np.array([1.2330601215362549, -1.2422520977309723, -1.6271378993988037, -2.3346992931761683, 0.3470206558704376,
        #      0.5055640339851379])

        # AIF exp
        # random
        # q0d = [1.1167820692062378, -1.3569369328073044, -1.7153658866882324, -1.8449112377562464, 1.3861088752746582, 0.8647269606590271]

        #near orthogonal (for testing) ~30cm from the finger
        # q0d = [0.9858736395835876, -1.4445417833379288, -1.8655898571014404, -1.713369985620016, 1.4113799333572388, 0.8625016212463379]

        # for testing control
        # q0d =  [0.48205387592315674, -1.1585588318160553, -1.9691475629806519, -1.6409017048277796, 1.8995541334152222, 0.21441252529621124]
        # q0d = [1.7740888595581055, -0.9471536439708252, -1.8968122005462646, -2.1335650883116664, 1.0458874702453613, 0.5070753693580627]
        # q0d = [1.387945532798767, -1.5586947363666077, -2.0203726291656494, -1.808474679986471, 1.1164849996566772, 2.0572235584259033]

        # previous good
        # q0d = [0.45207467675209045, -1.2947058540633698, -2.1917667388916016, -1.0930940073779603, 1.882698893547058, -1.516597096120016]

        q0d = [0.35332682728767395, -1.2308420699885865, -1.8594352006912231, -1.5463937253556033, 1.979645013809204, -1.5163329283343714]
        Q_now = Q[0:3, 0:3]

    # END OF IF

    if i == 2:
        q0d = [1.7740888595581055, -0.9471536439708252, -1.8968122005462646, -2.1335650883116664, 1.0458874702453613,
               0.5070753693580627]

        Q_now = Q[0:3, 0:3]
    # END OF IF

    # Move to the initial configuration
    rtde_c.moveJ(q0d, 0.5, 0.5)

    print('Waiting for hand tracking.')

    # Wait for a finger to appear
    while True:

        fingertips3d_result, isOcc = tracker.get_fingertips3d()


        time.sleep(0.002)
        # print(fingertips3d_result)
        if fingertips3d_result:
            break
        # END OF IF
    # END OF WHILE

    p_hat.shape = (3, 1)

    # after a finger is detected, wait for 3 seconds
    time.sleep(3)

    # make a beep sound for the humnan to start the demonstration
    beep_freq = 1500
    beep_dur = 1000
    winsound.Beep(beep_freq, beep_dur)
    # beep = lambda x: os.system("echo -n '\a';sleep 0.015;" * x)
    # beep(10)

    print('Hand is found!')

    # This variable is only for the anti-spike filter
    # p_prev = p_hat

    # log time
    tlog = np.array([t])


    # DMP initialization, with many default selections, e.g. Gaussian kernel, Linear Canonical system, alpha, beta ... 
    # M is the number of Kernels
    # dummy initialization of the DMP initial position and target
    # pT = p_hat
    x1 = np.zeros(3)
    if dmp_model.dmp_array[0].isTrained:
        x1 = dmp_model.p0
    else:
        dmp_model.set_goal(p_hat)  ## the initial trained the it is initialized in the loop

    x2 = np.zeros(3)
    x3 = 0


    # initialize time and discrete time
    t = 0
    k = 0

    print('Nk=', Nk)

    # Pass P as the new Q
    Q = np.array(P)



    # Uncomment if you want to free drive
    # rtde_c.freedriveMode()

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
        firstTime, pcf_hat, pcf_filtered, isOcc, isNan = get_pcf(firstTime, pcf_filtered)
        # print("pcf_filtered=", pcf_filtered)
        # print("isOcc=", isOcc)
        # print("pp=", pp)
        # print("Rp=", Rp)
        # print("pcf_filtered=", pcf_filtered)
        pcf_hat[2] = pcf_hat[2] + depth_calibration_param
        p_hat = pp + Rp @ pcf_hat
        print("p_hat=", p_hat)
        p_hat.shape = (3, 1)
        # initialize DMP state

        p_hat_iter = np.hstack((p_hat_iter, np.array(p_hat)))


        if (not dmp_model.dmp_array[0].isTrained) and (k==0):
            x1[0] = p_hat[0]
            x1[1] = p_hat[1]
            x1[2] = p_hat[2]


            dmp_model.set_init_state(x1)
        #END IF


        # get the Jacobian
        J = get_jacobian(ur, q)

        # shape arrays before stacking them
        #p_hat.shape = (3, 1)

        pp.shape = (3, 1)

        # Rp.shape = (3, 3)

        qp = rot2quat(Rp)

        qp.shape = (4, 1)


        # log time
        tlog = np.vstack((tlog, t))

        # compute Sigma now

        sigma_x_sq = pow(sigma_1/fx,2)*pow(sigma_3,2)+(pow(pcf_hat[0]/pcf_hat[2],2))*pow(sigma_3,2)+pow(pcf_hat[2]*sigma_1/fx,2)
        sigma_y_sq = pow(sigma_2/fy,2)*pow(sigma_3,2)+(pow(pcf_hat[1]/pcf_hat[2],2))*pow(sigma_3,2)+pow(pcf_hat[2]*sigma_2/fy,2)
        sigma_d_sq = pow(sigma_3,2)
        Sigma_1 = np.diag(np.array([sigma_x_sq, sigma_y_sq, sigma_d_sq]))
        if isOcc:
            print("Occluded! ---------------------------------------")
            Sigma_1 = 2 * Sigma_1
        Sigma_now = Rp  @ Sigma_1  @ np.transpose(Rp)



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

        #
        # ax2 = plt.axes(projection='3d')
        # Px, Py, Pz = getEllipsoidSurf(P[0:3, k*3:k*3+3], np.array([0, 0, 0]))
        # ax2.plot_surface(Px, Py, Pz, alpha=1.0, color='k')
        #
        # if i>0:
        #     Qx, Qy, Qz = getEllipsoidSurf(Q_now, np.array([0, 0, 0]))
        #     ax2.plot_surface(Qx, Qy, Qz, alpha=0.1, color='r')
        # # print('Sigma_now inv=', np.linalg.inv(Sigma_now))
        # Sigx, Sigy, Sigz = getEllipsoidSurf(Sigma_now, np.array([0, 0, 0]))
        # ax2.plot_surface(Sigx, Sigy, Sigz, alpha=0.1, color='b')
        #
        #
        # ax2.set_xlabel('x [m]')
        # ax2.set_ylabel('y [m]')
        # ax2.set_zlabel('z [m]')
        # ax2.set_aspect('equal', adjustable='box')
        # # plt.show()
        # plt.draw()
        # plt.pause(0.0001)
        # plt.clf()

        p_hat_temp = np.array(p_hat)
        p_hat_temp.shape = (1, 3)
        p_hat_temp = p_hat_temp[0]


        # ESTIMATION !!
        pstar = np.linalg.inv( Q_inv + Sigma_inv ) @ ( (Q_inv @ x1) + (Sigma_inv @ p_hat_temp) )



        # stack data array

        qp_iter = np.hstack((qp_iter, qp))
        pp_iter = np.hstack((pp_iter, np.array(pp)))
        pstar_log = np.hstack((pstar_log, np.array([ [pstar[0]], [pstar[1]], [pstar[2]] ])))
        x1_temp = np.array(x1)
        x1_temp.shape = (3,1)

        x1_log = np.hstack((x1_log, np.array(x1_temp)))

        ## ACtive perception signal
        detP = np.linalg.det(P[0:3, k*3:k*3+3])
        invP = np.linalg.inv(P[0:3, k*3:k*3+3])

        Jq = getJq(qp)

        Spp = skewSymmetric(pc-pp)

        A = P[0:3, k*3:k*3+3] @ P[0:3, k*3:k*3+3] @ Sigma_inv @ Sigma_inv

        ddet_dq = np.zeros(4)
        for j in range(4):

            dR_dqi = calculate_dR_d(qp, j)
            dSigma_dqi = dR_dqi @ Sigma_1  @ np.transpose(Rp) + Rp  @ Sigma_1  @ np.transpose(dR_dqi)
            dP_dqi = A @ dSigma_dqi
            ddet_dq[j] = detP * np.trace( invP @ dP_dqi )
        # END FOR

        v_p = np.zeros(6)
        v_p[3:6] = - ka * np.transpose(Jq) @ ddet_dq
        v_p[0:3] = Spp @ v_p[3:6]
        if pp[2]<0.20:
            v_p[2]=0.0

        print("v_p= ", v_p)
        # Inverse kinematics mapping with singularity avoidance
        qdot = np.linalg.pinv(J, 0.1) @ ( v_p )



        # Break if keyboard a is pressed
        if keyboard.is_pressed('a'):

            print('Stopping the iteration')

            break
            # END OF IF
        # END OF IF

        # set joint speed with acceleration limits
        # qdot = np.zeros(6)
        # =========================================
        rtde_c.speedJ(qdot, 1.0, dt)

        # print(Q[0:3, k * 3:k * 3 + 3])

        # print("k= ", k)
        k = k + 1


        # This is for synchronizing with the UR robot
        rtde_c.waitPeriod(t_start)

    # END OF WHILE -- control loop



    # Stop velocity control 
    rtde_c.speedStop()

    # Nk is the last index k
    Nk = k
    print("Nk = ", Nk)

    # make a sound (beep)
    beep_freq = 2000
    beep_dur = 500
    winsound.Beep(beep_freq, beep_dur)



    # Train the DMP model
    dmp_model.train(dt, p_hat_iter[:, 1:-1], plotPerformance = True)

    # Plots !-----------------
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(p_hat_iter[0, 1:-1], p_hat_iter[1, 1:-1], p_hat_iter[2, 1:-1])
    ax.plot3D(x1_log[0, 1:-1], x1_log[1, 1:-1], x1_log[2, 1:-1])
    ax.plot3D(pstar_log[0, 1:-1], pstar_log[1, 1:-1], pstar_log[2, 1:-1])
    ax.legend(['phat', 'x1', 'pstar'])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    # plot Ellipsoids -------------
    ax3 = plt.axes(projection='3d')
    k_test = Nk-1
    kk = 1
    while kk<Nk:

        cent = pstar_log[:,kk]
        cent.shape = (1,3)
        cent = cent[0]
        print(cent)
        Px, Py, Pz = getEllipsoidSurf(P[0:3, kk * 3:kk * 3 + 3], cent)
        ax3.plot_surface(Px, Py, Pz, alpha=1.0, color='k')

        if i > 0:
            Qx, Qy, Qz = getEllipsoidSurf(Q[0:3, kk * 3:kk * 3 + 3], cent )
            ax3.plot_surface(Qx, Qy, Qz, alpha=0.1, color='r')
        # print('Sigma_now inv=', np.linalg.inv(Sigma_now))
        Sigx, Sigy, Sigz = getEllipsoidSurf(Sigma_log[0:3, kk * 3:kk * 3 + 3], cent )
        ax3.plot_surface(Sigx, Sigy, Sigz, alpha=0.1, color='b')

        kk = kk + 500

    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_zlabel('z [m]')
    ax3.set_aspect('equal', adjustable='box')
    plt.show()
    # plt.draw()
    # plt.pause(0.0001)
    # plt.clf()

    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_zlabel('z [m]')
    ax3.set_aspect('equal', adjustable='box')
    plt.show()

    # write the log files
    data = {'phat': p_hat_iter, 'qp': qp_iter, 'pp': pp_iter, 't': tlog, 'P': P, 'Q': Q, 'Sigma':Sigma_log, 'pstar': pstar_log, 'x1': x1_log}
    scio.savemat('Logging_' + str(data_id) + '_' + str(i) + '.mat', data)


# END OF (BIG) FOR .... ITERATIONS
keyboard.unhook_all()

# stop the robot
rtde_c.speedStop()
rtde_c.stopScript()










