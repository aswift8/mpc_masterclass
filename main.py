import numpy as np

from run_sim import *

import matplotlib.pyplot as plt


### Parabolic trajectory (decelerating at 10m/s/s)

sim_time = 40
c_freq = 10
ts = np.arange(0, sim_time, 1/c_freq)

x = np.array([2000.0, -200.0, -10.0])

x_traj = x[0] + x[1] * ts - 0.5 * x[2] * np.multiply(ts, ts)
x_traj[(sim_time * c_freq)//2:] = 0

MPC = {
    'Np' : 10,
    'Nc' : 10,
    'rw' : 0.035
    }
PID = {
    'Kp' : 1,
    'Ki' : 0.2,
    'Kd' : 2,
    'N'  : 5
    }

#run_sim(sim_time, c_freq, x, x_traj, MPC, PID)


### Complex trajectory (shown in masterclass)

MPC = {
    'Np' : 20,
    'Nc' : 20,
    'rw' : 0.0035
    }
PID = {
    'Kp' : 1,
    'Ki' : 0.2,
    'Kd' : 2,
    'N'  : 5
    }

sim_time = 65
c_freq = 5
ts = np.arange(0, sim_time, 1/c_freq)

x = np.array([2573.49, -200.0, -10.0])

x_traj = np.zeros(ts.shape, dtype=float)
x_traj[0:10*c_freq] = 2573.49 - 200*ts[0:10*c_freq] + 0.5*10*ts[0:10*c_freq]*ts[0:10*c_freq]        # Decelerate at 10 m/s/s
x_traj[10*c_freq:29*c_freq] = 1073.49 - 100*ts[0:19*c_freq] + 0.5*5*ts[0:19*c_freq]*ts[0:19*c_freq] # Decelerate at 5 m/s/s
x_traj[29*c_freq:39*c_freq] = 75.99 - 5*ts[0:10*c_freq]                                             # Constant speed of 5 m/s
x_traj[39*c_freq:49*c_freq] = 25.99 - 5*ts[0:10*c_freq] + 0.5*0.5*ts[0:10*c_freq]*ts[0:10*c_freq]   # Decelerate at 0.5 m/s/s
x_traj[49*c_freq:sim_time*c_freq] = 1 - 0.1*ts[0:(sim_time-49)*c_freq]                              # Constant speed of 0.1 m/s until landing

plt.plot(ts, np.maximum(x_traj,0))
ylim = plt.gca().get_ylim()
for t in [0,10,29,39,48.8,58.8]:
    plt.plot([t,t], [-1e5,1e5], 'k--', linewidth=0.5)
plt.plot(ts, np.zeros(ts.shape), 'k--', linewidth=0.5)
plt.gca().set_ylim([0,ylim[1]])
plt.gca().set_xlim([0,sim_time])
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.show()

plt.plot(ts, np.maximum(x_traj,0))
ylim = plt.gca().get_ylim()
for t in [0,10,29,39,48.8,58.8]:
    plt.plot([t,t], [-1e5,1e5], 'k--', linewidth=0.5)
plt.plot(ts, np.zeros(ts.shape), 'k--', linewidth=0.5)
plt.gca().set_ylim([0,30])
plt.gca().set_xlim([39,sim_time])
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.show()

run_sim(sim_time, c_freq, x, x_traj, MPC, PID, [10,29,39,48.8,58.8])


### Sinusoidal trajectory

sim_time = 60
c_freq = 5
ts = np.arange(0, sim_time, 1/c_freq)

x = np.array([2000, 0, -10.0])

x_traj = x[0] + x[2]*(np.sin(ts/5))

#run_sim(sim_time, c_freq, x, x_traj, MPC, PID)
