import numpy as np

# A simple PID controller
class PID:
    def __init__(self, Kp, Ki, Kd, T, N):
        # Based on equations from
        # https://www.scilab.org/discrete-time-pid-controller-implementation
        a0 = 1 + N * T
        a1 = -(2 + N * T)
        a2 = 1
        b0 = Kp*a0 + Ki*T*a0 + Kd*N
        b1 = -(-Kp*a1 + Ki*T + 2*Kd*N)
        b2 = Kp + Kd*N
        self.cu1 = -a1/a0
        self.cu2 = -a2/a0
        self.ce0 = b0/a0
        self.ce1 = b1/a0
        self.ce2 = b2/a0

    def set_trajectory(self, x_ref):
        self.x_ref = x_ref
        self.i_ref = 0
        self.u1 = 0
        self.u2 = 0
        self.e1 = 0
        self.e2 = 0
        self.e = np.zeros(self.x_ref.shape)

    def step(self, x):
        # Update error tracker
        self.e[self.i_ref] = self.x_ref[self.i_ref] - x[0]
        # Perform PID
        # Calculate current error
        e0 = self.x_ref[self.i_ref] - x[0]
        # Calculate control response
        u = self.cu1 * self.u1 + self.cu2 * self.u2 + self.ce0 * e0 + \
            self.ce1 * self.e1 + self.ce2 * self.e2
        # Update stored controls and errors
        self.u2 = self.u1
        self.u1 = u
        self.e2 = self.e1
        self.e1 = e0
        # Update counter
        self.i_ref += 1
        # Return control
        return u

    def mse(self):
        e = self.e[:self.i_ref]
        return np.dot(e,e)/e.shape[0]
