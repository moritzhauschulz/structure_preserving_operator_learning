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

        self.trunk_data = self.trunk_data.view(self.trunk_data.shape[0], -1)
        #     self.labels = self.labels.view(self.labels.shape[0], -1)
        #     print('flattened data consolidating the last two dimensions')

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

    print(branch_data.shape)
    print(trunk_t.shape)
    print(y.shape)

    if args.method == 'deeponet':
        data = DeepOData(x, y.T)
    else:
        raise ValueError(f"Method {args.method} not implemented as data type.")

    return data

def enforce_hermitian_symmetry(u_hat):
    N = len(u_hat)
    # Ensure the DC component is real
    u_hat[0] = np.real(u_hat[0])
    
    if N % 2 == 0:
        # For even N, ensure Nyquist frequency is real
        u_hat[N//2] = np.real(u_hat[N//2])
        # Loop over frequencies 1 ... N//2 - 1
        for k in range(1, N//2):
            avg = 0.5 * (u_hat[k] + np.conj(u_hat[-k]))
            u_hat[k] = avg
            u_hat[-k] = np.conj(avg)
    else:
        # For odd N, loop over frequencies 1 ... (N-1)//2
        for k in range(1, (N+1)//2):
            avg = 0.5 * (u_hat[k] + np.conj(u_hat[-k]))
            u_hat[k] = avg
            u_hat[-k] = np.conj(avg)
    
    return u_hat

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

    # print("Before enforcement:", np.max(np.abs(u0_hat.imag)))

    # Ensure conjugate symmetry for real IFFT
    u0_hat = enforce_hermitian_symmetry(u0_hat)
    ut0_hat = enforce_hermitian_symmetry(ut0_hat)

    # print("After enforcement:", np.max(np.abs(u0_hat.imag)))

    return u0, ut0, u0_hat, ut0_hat, x_vals

def wave_evolve_fft(x_vals, t_vals, c):
    """Evolve the wave equation using FFT-based spectral solution."""
    Nx = len(x_vals)
    Nt = len(t_vals)    
    L = x_vals[-1] - x_vals[0]  # Domain length

    k = np.fft.fftfreq(Nx, d=(L/Nx)) * 2 * np.pi  # Wave numbers

    # Sample initial conditions
    u0, ut0, u0_hat, ut0_hat, x = wave_sample_initial_conditions_sinusoidal(x_vals, L)

    # Compute Fourier coefficients
    A_n = u0_hat
    B_n = np.zeros_like(A_n)
    
    nonzero_indices = np.abs(k) > 1e-10  
    B_n[nonzero_indices] = ut0_hat[nonzero_indices] / (c * k[nonzero_indices])
    B_n[0] = ut0_hat[0] / c  # Handle k=0 case correctly

    # Initialize storage
    energy_vals = []
    u_data = np.zeros((Nt, Nx), dtype=complex)
    ut_data = np.zeros((Nt, Nx), dtype=complex)

    for i, t in enumerate(t_vals):
        cos_term = np.cos(c * k * t)
        sin_term = np.sin(c * k * t)
        
        # Compute u(x,t) and u_t(x,t) in Fourier space
        u_hat_t = A_n * cos_term + B_n * sin_term
        ut_hat_t = -c * k * A_n * sin_term + c * k * B_n * cos_term

        # Transform back to physical space
        u_ifft = np.fft.ifft(u_hat_t)
        ut_ifft = np.fft.ifft(ut_hat_t)
        
        # Check imaginary part
        assert np.max(np.abs(u_ifft.imag)) < 1e-5, f"Warning: Nonzero imaginary part detected in u! {np.max(np.abs(u_ifft.imag))} at iteration {i}"
        assert np.max(np.abs(ut_ifft.imag)) < 1e-5, f"Warning: Nonzero imaginary part detected in ut! {np.max(np.abs(ut_ifft.imag))} at iteration {i}"

        u_data[i, :] = u_ifft.real
        ut_data[i, :] = ut_ifft.real

        # Compute energy
        energy_t = np.sum(np.abs(ut_hat_t)**2 + c**2 * np.abs(k * u_hat_t)**2) / Nx
        energy_vals.append(energy_t)

    return x, t_vals, u_data, ut_data, energy_vals, u0, ut0


def get_1d_wave_data(args):
    c = args.IC['c']
    t_vals = np.linspace(args.tmin, args.tmax, max(1,int((args.tmax-args.tmin)/args.t_res)))
    x_vals = np.linspace(args.xmin, args.xmax, args.col_N)

    Nx = len(x_vals)
    Nt = len(t_vals)

    if args.method == 'deeponet' or args.method == 'orthonormal_pushforward':
        branch_data = None
        trunk_t = None
        y = None
        for i in range(args.n_branch):
            x, t_vals, u_data, ut_data, energy_values, u0, ut0 = wave_evolve_fft(x_vals, t_vals, c)
            # print(f'energy values are {energy_values}')

            k = torch.tensor(np.fft.fftfreq(args.col_N, d=((args.xmax-args.xmin)/args.col_N)) * 2* np.pi).float()  # to device?

            branch_block = np.stack([u0,ut0], axis=0)
            branch_block = np.stack([branch_block] * t_vals.shape[0], axis=0)

            gt_u = torch.tensor(u0[None,:])
            gt_ut = torch.tensor(ut0[None,:])
            gt_u_hat = torch.fft.fft(gt_u, n=args.col_N, dim=1)  # dim=1 since shape is (time, space)
            gt_ut_hat = torch.fft.fft(gt_ut, n=args.col_N, dim=1)
            init_energy = torch.sum(torch.abs(gt_ut_hat)**2 + args.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) / args.col_N

            y_block = np.stack([u_data,ut_data], axis=0)
            y_block = np.stack([u_data, ut_data], axis=0).transpose(1, 0, 2)
            y_block = y_block.reshape(-1, 2, args.col_N)  # reshape to (time, 2, space)

            gt_u = torch.tensor(u_data)
            gt_ut = torch.tensor(ut_data)

            gt_u_hat = torch.fft.fft(gt_u, n=args.col_N, dim=1)  # dim=1 since shape is (time, space)
            gt_ut_hat = torch.fft.fft(gt_ut, n=args.col_N, dim=1)
            target_energy = torch.sum(torch.abs(gt_ut_hat)**2 + args.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) / args.col_N

            #check init energy approximately equal to average target energy
            assert init_energy[0] - torch.mean(target_energy) < 1e-1, f'Initial energy not equal to average target energy. {init_energy[0]} != {torch.mean(target_energy)}'

            if branch_data is None:
                branch_data = branch_block
                trunk_t = t_vals
                y = y_block
            else:
                branch_data = np.concatenate((branch_data, branch_block), axis=0)
                trunk_t = np.concatenate((trunk_t, t_vals), axis=0)
                y = np.concatenate((y, y_block), axis=0)

        x = (branch_data.astype(np.float32), trunk_t.astype(np.float32))
        y = y.astype(np.float32)

        #init energy
        gt_u = torch.tensor(x[0][:,0,:])
        gt_ut = torch.tensor(x[0][:,1,:])

        gt_u_hat = torch.fft.fft(gt_u, n=args.col_N, dim=1)  # dim=1 since shape is (time, space)
        gt_ut_hat = torch.fft.fft(gt_ut, n=args.col_N, dim=1)
        true_energy = torch.sum(torch.abs(gt_ut_hat)**2 + args.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) / args.col_N

        #target energy
        gt_u = torch.tensor(y[:,0,:])
        gt_ut = torch.tensor(y[:,1,:])

        gt_u_hat = torch.fft.fft(gt_u, n=args.col_N, dim=1)  # dim=1 since shape is (time, space)
        gt_ut_hat = torch.fft.fft(gt_ut, n=args.col_N, dim=1)
        target_energy = torch.sum(torch.abs(gt_ut_hat)**2 + args.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) / args.col_N

        assert (true_energy - target_energy).max() < 1, f'Energy is not conserved in training data – max difference was: {(true_energy - target_energy).max()}'

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
