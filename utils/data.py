import os
import pickle
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset

def get_data(args):
    print('getting data')
    print(args.load_data)
    if args.load_data and os.path.exists(args.data_dir + args.data_config):
        with open(args.data_dir + args.data_config, 'rb') as f:
            data = pickle.load(f)
    elif args.method == 'deeponet':
        if args.problem == 'harmonic_oscillator':
            assert len(args.IC) == 3, 'Initial conditions for harmonic oscillator must be a list of length 3.'
            data = [get_harmonic_oscillator_data(args), get_harmonic_oscillator_data(args)]
        elif args.problem == '1d_KdV_Soliton':
            assert len(args.IC) == 2, 'Initial conditions for 1d-KdV with Soliton must be a list of length 2.'
            if args.use_ifft:
                data = [get_1d_KdV_Soliton_data_ifft(args), get_1d_KdV_Soliton_data_ifft(args)]
            else:
                data = [get_1d_KdV_Soliton_data(args), get_1d_KdV_Soliton_data(args)]
        else:
            raise ValueError(f"Problem {args.problem} not recognized.")
    else:
        raise ValueError(f"Method {args.method} not recognized.")
    if args.save_data:
        with open(args.data_dir + args.data_config, 'wb') as f:
            pickle.dump(data, f)
    return data

class DeepOData(Dataset):
    def __init__(self, x, y):
        self.branch_data = torch.tensor(x[0])
        self.trunk_data = torch.tensor(x[1])
        self.labels = torch.tensor(y).T

    def __len__(self):
        return len(self.branch_data)  # Total size of the dataset

    def __getitem__(self, idx):
        return (self.branch_data[idx], self.trunk_data[idx]), self.labels[idx]  # Return a single data-label pair

# Define the ODE system
def harmonic_oscillator(t, z, omega):
    x, v = z  # z = [x, v], where v = dx/dt
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

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

    if args.num_outputs == 1:
        y = y[0:1, :]  # Retain only the first variable in the solution matrix y

    trunk_data = np.tile(trunk_data, (1, args.n_branch)).transpose() #n_trunk * n_trunk
    branch_data = np.repeat(branch_data, repeats=args.n_trunk, axis=0)

    x = (branch_data.astype(np.float32), trunk_data.astype(np.float32))
    y = y.astype(np.float32)

    if args.method == 'deeponet':
        data = DeepOData(x, y)
    else:
        raise ValueError(f"Method {args.method} not implemented as data type.")

    return data

# Make 1d_KdV_Soliton dataset

def exact_soliton(x, t, c, a):
    if isinstance(x, torch.Tensor):
        arg = torch.clamp(torch.sqrt(c) * (x - c * t - a) / 2, -50, 50)  # Prevent extreme values
        return ((c / 2) / torch.cosh(arg) ** 2)  # Stable sech^2 computation
    else:
        arg = np.clip(np.sqrt(c) * (x - c * t - a) / 2, -50, 50)  # Prevent extreme values
        return ((c / 2) / np.cosh(arg) ** 2)  # Stable sech^2 computation


def get_1d_KdV_Soliton_data(args):
    a = np.random.uniform(args.IC['a'][0], args.IC['a'][1], size=(args.n_branch, 1))  # parameter 1
    c = np.random.uniform(args.IC['c'][0], args.IC['c'][1], size=(args.n_branch, 1))  # parameter 2
    trunk_t =  np.linspace(args.tmin, args.tmax, int((args.tmax-args.tmin)/args.t_res))
    trunk_x = np.linspace(args.xmin, args.xmax, int((args.xmax-args.xmin)/args.col_N))

    X, T = np.meshgrid(trunk_x, trunk_t)
    trunk_data = np.column_stack((T.flatten(), X.flatten()))

    branch_data = np.concatenate((a,c), axis=1) #n_branch x 2

    # Generate solution for each parameter set
    y = np.array([exact_soliton(trunk_data[:,1], trunk_data[:,0], c_i, a_i) 
                  for a_i, c_i in zip(branch_data[:,0], branch_data[:,1])])
    y = y.reshape(-1, 1).T  # Reshape to match expected format
    y = y.astype(np.float32)

    branch_data = np.repeat(branch_data, repeats=trunk_data.shape[0], axis=0)
    trunk_data = np.tile(trunk_data, (args.n_branch, 1))

    x = (branch_data.astype(np.float32), trunk_data.astype(np.float32))

    if args.method == 'deeponet':
        data = DeepOData(x, y)
    else:
        raise ValueError(f"Method {args.method} not implemented as data type.")

    return data

def get_1d_KdV_Soliton_data_ifft(args):
    a = np.random.uniform(args.IC['a'][0], args.IC['a'][1], size=(args.n_branch, 1))  # parameter 1
    c = np.random.uniform(args.IC['c'][0], args.IC['c'][1], size=(args.n_branch, 1))  # parameter 2
    trunk_t =  np.linspace(args.tmin, args.tmax, max(1,int((args.tmax-args.tmin)/args.t_res)))
    trunk_x = np.linspace(args.xmin, args.xmax, args.col_N)


    # X, T = np.meshgrid(trunk_x, trunk_t)
    # trunk_data = np.column_stack((T.flatten(), X.flatten()))

    branch_data = np.concatenate((a,c), axis=1) #n_branch x 2
    branch_data = np.repeat(branch_data, repeats=trunk_t.shape[0], axis=0)
    trunk_t = np.tile(trunk_t, (1, args.n_branch)).T
    # Generate solution for each parameter set
    y = np.array([exact_soliton(trunk_x, t, c_i, a_i) 
                  for a_i, c_i, t in zip(branch_data[:,0], branch_data[:,1], trunk_t)])
    # y = y.reshape(-1, 1).T  # Reshape to match expected format
    y = y.astype(np.float32).T

    # trunk_data = trunk_t
    # trunk_data = np.tile(trunk_data, (1, args.n_branch)).T

    x = (branch_data.astype(np.float32), trunk_t.astype(np.float32))

    if args.method == 'deeponet':
        data = DeepOData(x, y)
    else:
        raise ValueError(f"Method {args.method} not implemented as data type.")

    return data

def get_1d_KdV_Soliton_data_ifft(args):
    a = np.random.uniform(args.IC['a'][0], args.IC['a'][1], size=(args.n_branch, 1))  # parameter 1
    c = np.random.uniform(args.IC['c'][0], args.IC['c'][1], size=(args.n_branch, 1))  # parameter 2
    trunk_t =  np.linspace(args.tmin, args.tmax, max(1,int((args.tmax-args.tmin)/args.t_res)))
    trunk_x = np.linspace(args.xmin, args.xmax, args.col_N)


    # X, T = np.meshgrid(trunk_x, trunk_t)
    # trunk_data = np.column_stack((T.flatten(), X.flatten()))

    branch_data = np.concatenate((a,c), axis=1) #n_branch x 2
    branch_data = np.repeat(branch_data, repeats=trunk_t.shape[0], axis=0)
    trunk_t = np.tile(trunk_t, (1, args.n_branch)).T
    # Generate solution for each parameter set
    y = np.array([exact_soliton(trunk_x, t, c_i, a_i) 
                  for a_i, c_i, t in zip(branch_data[:,0], branch_data[:,1], trunk_t)])
    # y = y.reshape(-1, 1).T  # Reshape to match expected format
    y = y.astype(np.float32).T

    # trunk_data = trunk_t
    # trunk_data = np.tile(trunk_data, (1, args.n_branch)).T

    x = (branch_data.astype(np.float32), trunk_t.astype(np.float32))

    if args.method == 'deeponet':
        data = DeepOData(x, y)
    else:
        raise ValueError(f"Method {args.method} not implemented as data type.")

    return data
