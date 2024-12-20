import os
import pickle
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset

def get_data(args):
    if args.load_data and os.path.exists(args.data_dir + args.data_config):
        with open(args.data_dir + args.data_config, 'rb') as f:
            data = pickle.load(f)
    elif args.method == 'deeponet':
        if args.problem == 'harmonic_oscillator':
            assert len(args.IC) == 3, 'Initial conditions for harmonic oscillator must be a list of length 3.'
            data = [get_harmonic_oscillator_data(args), get_harmonic_oscillator_data(args)]
        else:
            raise ValueError(f"Problem {args.problem} not recognized.")
    else:
        raise ValueError(f"Method {args.method} not recognized.")
    if args.save_data:
        with open(args.data_dir + args.data_config, 'wb') as f:
            pickle.dump(data, f)
    return data

# Define the ODE system
def harmonic_oscillator(t, z, omega):
    x, v = z  # z = [x, v], where v = dx/dt
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

class DeepOData(Dataset):
    def __init__(self, x, y):
        self.branch_data = torch.tensor(x[0])
        self.trunk_data = torch.tensor(x[1])
        self.labels = torch.tensor(y).T

    def __len__(self):
        return len(self.branch_data)  # Total size of the dataset

    def __getitem__(self, idx):
        return (self.branch_data[idx], self.trunk_data[idx]), self.labels[idx]  # Return a single data-label pair


# Make harmonic oscillator dataset
def get_harmonic_oscillator_data(args):
    q0 = np.random.uniform(args.IC['q0'][0], args.IC['q0'][1], size=(args.n_branch, 1))  # Initial position (x(0))
    p0 = np.random.uniform(args.IC['p0'][0], args.IC['p0'][1], size=(args.n_branch, 1))  # Initial velocity (dx/dt at t=0)
    omega = np.random.uniform(args.IC['omega'][0], args.IC['omega'][1], size=(args.n_branch, 1)) # Angular frequency
    trunk_data =  np.linspace(args.tmin, args.tmax, args.n_trunk)

    branch_data = np.concatenate((q0,p0,omega), axis=1) #n_branch x 3
    t_span = (0, args.tmax)

    #generate y
    y = np.array([[], []])
    for i in range(args.n_branch):
        # Initial conditions
        z0 = branch_data[i,0:2] #initial phase space
        (omega) = branch_data[i,2] #angular velocity
        # Solve the ODE
        solution = solve_ivp(harmonic_oscillator, t_span, z0, args=(omega,), t_eval=trunk_data)

        y = np.concatenate([y, solution.y], axis=1)

    trunk_data = np.tile(trunk_data, (1, args.n_branch)).transpose() #n_trunk * n_trunk
    branch_data = np.repeat(branch_data, repeats=args.n_trunk, axis=0)

    x = (branch_data.astype(np.float32), trunk_data.astype(np.float32))
    y = y.astype(np.float32)

    if args.method == 'deeponet':
        data = DeepOData(x, y)
    else:
        raise ValueError(f"Method {args.method} not implemented as data type.")

    return data

