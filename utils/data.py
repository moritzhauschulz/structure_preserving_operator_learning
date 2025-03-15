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
        elif args.problem == '1d_wave':
            assert len(args.IC) == 1, 'Initial conditions for 1d wave must be a list of length 1.'
            data = [get_1d_wave_data(args), get_1d_wave_data(args)]
        else:
            raise ValueError(f"Problem {args.problem} not recognized.")
    elif args.method == 'full_fourier':
        if args.problem == '1d_wave':
            assert len(args.IC) == 1, 'Initial conditions for 1d wave must be a list of length 1.'
            data = [get_1d_wave_data(args), get_1d_wave_data(args)]
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
        self.labels = torch.tensor(y)

        if len(self.branch_data.shape) > 2:
            self.branch_data = self.branch_data.view(self.branch_data.shape[0], -1)
            self.trunk_data = self.trunk_data.view(self.trunk_data.shape[0], -1)
            self.labels = self.labels.view(self.labels.shape[0], -1)
            print('flattened data consolidating the last two dimensions')

    def __len__(self):
        return len(self.branch_data)  # Total size of the dataset

    def __getitem__(self, idx):
        return (self.branch_data[idx], self.trunk_data[idx]), self.labels[idx]  # Return a single data-label pair

class SpectralSpaceTime(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(x)

    def __len__(self):
        return len(self.x.shape[0])  # Total size of the dataset

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]  # Return a single data-label pair


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
    print(y.shape)

    # trunk_data = trunk_t
    # trunk_data = np.tile(trunk_data, (1, args.n_branch)).T

    x = (branch_data.astype(np.float32), trunk_t.astype(np.float32))

    if args.method == 'deeponet':
        data = DeepOData(x, y)
    else:
        raise ValueError(f"Method {args.method} not implemented as data type.")

    return data

def wave_sample_initial_conditions_sinusoidal(x_vals, L):
    """Generate smooth initial conditions as a sum of sinusoids, then transform to Fourier space."""
    Nx = len(x_vals)

    # Create initial condition as sum of sinusoids
    k_max = 10  
    amplitudes = np.random.randn(k_max) * np.exp(-np.linspace(0, k_max, k_max)**2 / 10)
    phases = np.random.uniform(0, 2 * np.pi, k_max)
    
    u0 = np.sum([a * np.sin(k * np.pi * x_vals / (L/2) + p) for k, a, p in zip(range(1, k_max + 1), amplitudes, phases)], axis=0)
    ut0 = np.sum([a * np.cos(k * np.pi * x_vals / (L/2) + p) for k, a, p in zip(range(1, k_max + 1), amplitudes, phases)], axis=0)

    # Convert to Fourier space
    u0_hat = np.fft.fft(u0)
    ut0_hat = np.fft.fft(ut0)

    # Ensure conjugate symmetry for real IFFT
    u0_hat[Nx//2 + 1:] = np.conj(np.flip(u0_hat[:Nx//2]))
    ut0_hat[Nx//2 + 1:] = np.conj(np.flip(ut0_hat[:Nx//2]))

    return u0, ut0, u0_hat, ut0_hat, x_vals

def wave_evolve_fft(x_vals,t_vals, c):
    """Evolve the wave equation using FFT-based spectral solution."""
    Nx = len(x_vals)
    Nt = len(t_vals)    
    L = x_vals[-1] - x_vals[0]  # Domain length


    k = np.fft.fftfreq(Nx, d=(L/Nx)) * 2 * np.pi  # Wave numbers

    # Sample smooth initial conditions
    u0, ut0, u0_hat, ut0_hat, x = wave_sample_initial_conditions_sinusoidal(x_vals, L)

    # Compute Fourier coefficients for evolution
    A_n = u0_hat
    B_n = np.zeros_like(A_n)
    nonzero_indices = np.abs(k) > 1e-10  
    B_n[nonzero_indices] = ut0_hat[nonzero_indices] / (c * k[nonzero_indices])

    # Initialize storage
    energy_vals = []
    u_data = np.zeros((Nt, Nx))
    ut_data = np.zeros((Nt, Nx))

    for i, t in enumerate(t_vals):
        cos_term = np.cos(c * k * t)
        sin_term = np.sin(c * k * t)
        
        # Compute u(x,t) and u_t(x,t) in Fourier space
        u_hat_t = A_n * cos_term + B_n * sin_term
        ut_hat_t = -c * k * A_n * sin_term + c * k * B_n * cos_term

        # Transform back to physical space
        u_data[i, :] = np.fft.ifft(u_hat_t).real
        ut_data[i, :] = np.fft.ifft(ut_hat_t).real

        # Compute energy using Parseval’s theorem
        energy_t = np.sum(np.abs(ut_hat_t)**2 + c**2 * np.abs(k * u_hat_t)**2) / Nx
        energy_vals.append(energy_t)

    print(u_data.shape)

    return x, t_vals, u_data, ut_data, energy_vals, u0, ut0

def get_1d_wave_data(args):
    c = args.IC['c']
    t_vals = np.linspace(args.tmin, args.tmax, max(1,int((args.tmax-args.tmin)/args.t_res)))
    x_vals = np.linspace(args.xmin, args.xmax, args.col_N)

    Nx = len(x_vals)
    Nt = len(t_vals)

    if args.num_output_fn == 1:
        print('Only targeting u(t,x) and not derivative. Increase num_pure_output_fn to target both.')

    if args.method == 'deeponet' or args.method == 'orthonormal_pushforward':
        branch_data = None
        trunk_t = None
        y = None
        for i in range(args.n_branch):
            x, t_vals, u_data, ut_data, energy_values, u0, ut0 = wave_evolve_fft(x_vals, t_vals, c)

            branch_block = np.concatenate((u0.reshape(-1, 1), ut0.reshape(-1, 1)), axis=1).reshape(1,2,-1) #n_branch x 2 x num_x
            branch_block = np.repeat(branch_block, repeats=t_vals.shape[0], axis=0)
            if args.num_output_fn == 1:
                y_block = u_data.reshape(-1,1,Nx)
            else:
                y_block = np.concatenate((u_data,ut_data), axis=0).reshape(-1,2,Nx) #n_branch x 2 x num_x 
            # y_block = y_block.reshape(-1, 1).reshape(1,2,-1) 
            if branch_data is None:
                branch_data = branch_block
                trunk_t = t_vals
                y = y_block
            else:
                branch_data = np.concatenate((branch_data, branch_block), axis=0)
                trunk_t = np.concatenate((trunk_t, t_vals), axis=0)
                y = np.concatenate((y, y_block), axis=0)
        
        y = y.astype(np.float32)

        x = (branch_data.astype(np.float32), trunk_t.astype(np.float32))
        data = DeepOData(x, y)
    elif args.method == 'full_fourier':
        x = None
        y = None
        for i in range(args.n_branch):
            x_vals, t_vals, u_data, ut_data, energy_values, u0, ut0 = wave_evolve_fft(x_vals, t_vals, c)
            x_block = np.concatenate((u0.reshape(-1, 1), ut0.reshape(-1, 1)), axis=1).reshape(1,2,-1) #n_branch x 2 x num_x
            y_block = np.concatenate((u_data.reshape(1,Nt,Nx),ut_data.reshape(1,Nt,Nx)), axis=0).reshape(1,2,Nt,Nx) #n_branch x 2 x num_x 
            # y_block = y_block.reshape(-1, 1).reshape(1,2,-1) 
            if x is None:
                x = x_block
                y = y_block
            else:
                x = np.concatenate((x, x_block), axis=0)
                y = np.concatenate((y, y_block), axis=0)

        data = SpectralSpaceTime(x, y)
    else:
        raise ValueError(f"Method {args.method} not implemented as data type.")

    return data
