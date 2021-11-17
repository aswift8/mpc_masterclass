import numpy as np

# A Model Predictive Controller designed for a single-input, single-output, linear problem
class MPC:
    def __init__(self, A, B, C, Np, Nc, rw):
        self.Np = Np
        # Construct F, Phi matrices
        # Y = F x + Phi U
        # Y   - future states
        # F   - project from initial state
        # Phi - changes from control
        F = np.zeros(shape=(Np, 3))
        Phi = np.zeros(shape=(Np, Nc))
        for p in range(0, Np):
            v = C
            for i in range(0, p + 1):
                v = np.dot(v, A)
            F[p,:] = v

            for i in range(0,Np):
                v = np.identity(3)
                for j in range(i):
                    v = np.dot(v, A)
                v = np.dot(np.dot(C, v), B)
                for j in range(0, Nc):
                    if i + j < Np:
                        Phi[i+j, j] = v
        self.F = F
        self.Phi = Phi
        # Cost function
        # J = (Ref - Y)' (Ref - Y) + U' Rw U
        # Find U that minimises J
        # Analytical solution: dJ/dU = 0
        # U = inv(Phi' Phi - Rw) Phi' (Ref - Fx)
        # Phi, Rw, Fx are constants
        # U = K (Ref - Fx)
        self.K = np.dot(np.linalg.inv(np.dot(Phi.T, Phi) + np.identity(Nc) * rw), Phi.T)

    def set_trajectory(self, x_ref):
        # Linearly extrapolate trajectory Np steps
        self.x_ref = np.append(x_ref, x_ref[-1]+(x_ref[-1]-x_ref[-2])*np.arange(1,self.Np+1))
        self.i_ref = 1
        self.e = np.zeros(self.x_ref.shape)

    def step(self, x):
        # Update error tracker
        self.e[self.i_ref-1] = self.x_ref[self.i_ref-1] - x[0]
        # Perform MPC
        # Get reference from trajectory for current step
        Ref = self.x_ref[self.i_ref:self.i_ref+self.Np]
        # Predict future states
        Fx = np.dot(self.F, x)
        # Get error between reference and projection
        Ref_diff = Ref - Fx
        # Calculate control vector
        U = np.dot(self.K, Ref_diff)
        self.i_ref += 1
        # Apply first control step
        return U[0]

    def mse(self):
        e = self.e[:self.i_ref]
        return np.dot(e,e)/e.shape[0]
