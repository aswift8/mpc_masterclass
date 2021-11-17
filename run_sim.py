import numpy as np
import matplotlib.pyplot as plt

from simulation import Simulation
import mpc
import pid


# Helper method to allow for quick plotting of different problems
def run_sim(sim_time, c_freq, x, x_traj, MPC, PID, time_marks=[]):
    sim_steps = sim_time * c_freq
    dt = 1/c_freq

    ts = np.arange(0, sim_time, dt)

    print("Creating MPC controller")
    A = np.array([
        [1, dt,  0],
        [0,  1, dt],
        [0,  0,  1]], dtype=float)
    B = np.array([
        [ 0],
        [dt],
        [ 0]], dtype=float)
    C = np.array([1, 0, 0], dtype=float)
    controller_MPC = mpc.MPC(A, B, C, MPC['Np'], MPC['Nc'], MPC['rw'])

    print("Creating PID controller")
    controller_PID = pid.PID(PID['Kp'], PID['Ki'], PID['Kd'], dt, PID['N'])
    
    print("Creating simulations")
    sim_MPC = Simulation(x)
    sim_PID = Simulation(x)

    controller_MPC.set_trajectory(x_traj)
    controller_PID.set_trajectory(x_traj)

    print("Running MPC simulation")
    ts, ys_MPC, us_MPC = sim_MPC.run(sim_steps, controller_MPC, c_freq, round(10000/c_freq))
    print(f"MPC MSE: {controller_MPC.mse()}, final height: {ys_MPC[-1]}")
    print("Running PID simulation")
    _ , ys_PID, us_PID = sim_PID.run(sim_steps, controller_PID, c_freq, round(10000/c_freq))
    print(f"PID MSE: {controller_PID.mse()}, final height: {ys_PID[-1]}")

    print("Plotting results")

    x_traj = np.maximum(x_traj, 0)
    """
    # Paths
    plt.subplot(2,1,1)
    plt.plot(ts, ys_MPC, linewidth=0.5)
    plt.plot(ts, ys_PID, linewidth=0.5)
    plt.plot(ts, x_traj, linewidth=0.5)
    ylim = plt.gca().get_ylim()
    for t in time_marks:
        plt.plot([t,t], [-1e5,1e5], 'k--', linewidth=0.5)
    plt.gca().set_ylim(ylim)
    #plt.plot(ts, np.zeros(ts.shape), 'k--', linewidth=0.5)
    """

    # Errors
    plt.subplot(2,1,1)
    es_MPC = x_traj - ys_MPC.flatten()
    es_PID = x_traj - ys_PID.flatten()
    plt.plot(ts, es_MPC, linewidth=0.5)
    plt.plot(ts, es_PID, linewidth=0.5)
    plt.plot(ts, np.zeros(ts.shape), 'k--', linewidth=0.5)
    ylim = plt.gca().get_ylim()
    for t in time_marks:
        plt.plot([t,t], [-1e5,1e5], 'k--', linewidth=0.5)
    plt.gca().set_ylim(ylim)
    plt.gca().set_xlim([0,sim_time])
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.legend(["Custom","PID"])

    # Control
    plt.subplot(2,1,2)
    
    tus = np.empty((2 * ts.size,), dtype=ts.dtype)
    tus[0::2] = ts
    tus[1::2] = ts + dt
    uus_MPC = np.empty((2 * us_MPC.size,1), dtype=us_MPC.dtype)
    uus_MPC[0::2] = us_MPC
    uus_MPC[1::2] = us_MPC
    uus_PID = np.empty((2 * us_PID.size,1), dtype=us_PID.dtype)
    uus_PID[0::2] = us_PID
    uus_PID[1::2] = us_PID
    plt.plot(tus, uus_MPC, linewidth=0.5)
    plt.plot(tus, uus_PID, linewidth=0.5)
    plt.plot(ts, np.zeros(ts.shape), 'k--', linewidth=0.5)
    ylim = plt.gca().get_ylim()
    for t in time_marks:
        plt.plot([t,t], [-1e5,1e5], 'k--', linewidth=0.5)
    plt.gca().set_ylim(ylim)
    plt.gca().set_xlim([0,sim_time])
    plt.xlabel("Time (s)")
    plt.ylabel("Control (m/$s^2$)")
    plt.legend(["Custom","PID"])

    plt.show()
