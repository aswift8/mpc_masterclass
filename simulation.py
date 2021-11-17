import numpy as np

class Simulation:

    def __init__(self, x):
        # x = [position, velocity, acceleration]'
        self.x = x.copy()

    def run(self, iterations, controller, controller_freq, simulation_timestep_mult):
        dt_con = 1 / controller_freq
        dt_sim = dt_con / simulation_timestep_mult

        # Initialize return values
        ts = np.arange(0, iterations*dt_con, dt_con)
        ys = np.zeros(shape=(iterations, 1), dtype=float)
        us = np.zeros(shape=(iterations, 1), dtype=float)

        crash_flag = False      # True if we just hit the ground
        # Run through simulation until finished or crashed
        for i in range(iterations):
            if crash_flag:
                break
            ys[i] = self.x[0]				# Update position tracking
            u = controller.step(self.x)		# Get control from controller
            u = max(min(u, 30), 0)			# Apply constraints
            us[i] = u						# Update control tracking
            for j in range(simulation_timestep_mult):
                # For each iteration, separate movement into smaller timesteps to increase accuracy
                # Euler integration
                self.x[1] += dt_sim * (self.x[2] + u)
                self.x[0] += dt_sim * self.x[1]
                # Check for crash
                if self.x[0] < 0:
                    self.x[0] = 0
                    print(f"Hit ground at {self.x[1]}m/s, t={i*dt_con + j*dt_sim}s")
                    crash_flag = True
                    break
        return ts,ys,us
