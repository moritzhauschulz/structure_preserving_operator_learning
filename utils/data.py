import os
import pickle
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def get_data(args):
    print('getting data')
    if args.load_data and os.path.exists(args.data_dir + args.data_config) and args.method == 'full_fourier':
        print(f'loading data')
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
            # assert len(args.IC) == 1, 'Initial conditions for 1d wave must be a list of length 1.'
            data = [get_1d_wave_data(args), get_1d_wave_data(args)]
        else:
            raise ValueError(f"Problem {args.problem} not recognized.")
    elif args.method == 'full_fourier':
        if args.problem == '1d_wave':
            # assert len(args.IC) == 1, 'Initial conditions for 1d wave must be a list of length 1.'
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
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)  # Total size of the dataset

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
        data = DeepOData(x, y.T)
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
    trunk_t =  np.linspace(args.tmin, args.tmax, int((args.tmax-args.tmin)/args.t_res), endpoint=False)
    trunk_x = np.linspace(args.xmin, args.xmax, int((args.xmax-args.xmin)/args.Nx), endpoint=False)

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
    trunk_t =  np.linspace(args.tmin, args.tmax, max(1,int((args.tmax-args.tmin)/args.t_res)), endpoint=False)
    trunk_x = np.linspace(args.xmin, args.xmax, args.Nx, endpoint=False)


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


def wave_sample_initial_conditions_periodic_gp(x_vals, L, num_modes, zero_zero_mode=True,lengthscale=1.0, variance=1.0, period=None, noise=1e-6):
    """
    Generate smooth initial conditions using a periodic Gaussian process.

    Parameters:
        x_vals (numpy array): Spatial grid points.
        L (float): Domain length.
        lengthscale (float): Controls smoothness.
        variance (float): Controls amplitude of variations.
        period (float, optional): Period of the GP. If None, set to L.
        noise (float): Small noise term for numerical stability.

    Returns:
        u0 (numpy array): Initial condition in physical space.
        ut0 (numpy array): Time derivative of initial condition.
        u0_hat (numpy array): Fourier transform of u0.
        ut0_hat (numpy array): Fourier transform of ut0.
        x_vals (numpy array): Spatial grid points.
    """
    Nx = len(x_vals)
    x_vals_modes = np.linspace(x_vals[0], x_vals[-1], num_modes)  # Grid points for GP


    # Set period to domain length if not provided
    if period is None:
        period = L

    # Define the periodic kernel
    def periodic_kernel(x1, x2, lengthscale, variance, period):
        dist = np.abs(np.subtract.outer(x1, x2))  # Compute |x - x'|
        return variance * np.exp(-2 * (np.sin(np.pi * dist / period) ** 2) / lengthscale**2)

    # Compute covariance matrix
    K = periodic_kernel(x_vals_modes, x_vals_modes, lengthscale, variance, period)
    K += noise * np.eye(num_modes)  # Add small noise for numerical stability

    # Sample from the GP prior
    u0 = np.random.multivariate_normal(mean=np.zeros(num_modes), cov=K)
    ut0 = np.random.multivariate_normal(mean=np.zeros(num_modes), cov=K)  # Independent sample for velocity

    # Convert to Fourier space
    u0_hat = np.fft.fft(u0)
    ut0_hat = np.fft.fft(ut0)

    #zero out the zero-th mode
    if zero_zero_mode:
        u0_hat[0] = 0
        ut0_hat[0] = 0

    # Ensure conjugate symmetry for real IFFT using provided function
    u0_hat = enforce_hermitian_symmetry(u0_hat)
    ut0_hat = enforce_hermitian_symmetry(ut0_hat)

    # Pad higher modes with zeros
    u0_hat_padded = np.zeros(Nx, dtype=complex)
    ut0_hat_padded = np.zeros(Nx, dtype=complex)

    # Copy non-zero modes
    u0_hat_padded[:num_modes//2] = u0_hat[:num_modes//2]
    u0_hat_padded[-num_modes//2:] = u0_hat[-num_modes//2:]
    ut0_hat_padded[:num_modes//2] = ut0_hat[:num_modes//2]
    ut0_hat_padded[-num_modes//2:] = ut0_hat[-num_modes//2:]

    # Transform back to physical space with padded spectrum
    u0 = np.fft.ifft(u0_hat_padded).real * Nx/num_modes
    ut0 = np.fft.ifft(ut0_hat_padded).real * Nx/num_modes

    # Update Fourier coefficients
    u0_hat = np.fft.fft(u0) 
    ut0_hat = np.fft.fft(ut0)

    # #check periodicity
    # assert np.allclose(u0[0], u0[-1]), f'Initial condition not periodic. u0[0] = {u0[0]} != {u0[-1]} = u0[-1]'

    return u0, ut0, u0_hat, ut0_hat, x_vals


def wave_sample_initial_conditions_sinusoidal(x_vals, L):
    """Generate smooth initial conditions as a sum of sinusoids, then transform to Fourier space."""
    Nx = len(x_vals)

    # Create initial condition as sum of sinusoids
    k_max = 10  
    amplitudes = np.random.randn(k_max) * np.exp(-np.linspace(0, k_max, k_max, endpoint=False)**2 / 10)
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

    #recalculate u0 and ut0
    u0 = np.fft.ifft(u0_hat).real
    ut0 = np.fft.ifft(ut0_hat).real

    # print("After enforcement:", np.max(np.abs(u0_hat.imag)))

    return u0, ut0, u0_hat, ut0_hat, x_vals

def wave_evolve_fft(args, x_vals, t_vals, c):
    """Evolve the wave equation using a high-accuracy numerical spectral method."""
    Nx = len(x_vals)
    Nt = len(t_vals)    
    L = args.xmax - args.xmin  # Domain length
    
    # Use smaller timestep for integration
    dt = args.data_dt
    n_substeps = max(1, int((t_vals[1] - t_vals[0])/dt))
    dt = (t_vals[1] - t_vals[0])/n_substeps
    t_fine = torch.arange(t_vals[0], t_vals[-1] + dt/2 + dt, dt)

    k = torch.fft.fftfreq(Nx, d=(L/Nx)) * 2 * torch.pi  # Wave numbers

    # Sample initial conditions
    if args.IC['type'] == 'periodic_gp':
        u0, ut0, u0_hat, ut0_hat, x = wave_sample_initial_conditions_periodic_gp(x_vals, L, args.data_modes, zero_zero_mode=args.zero_zero_mode, **args.IC['params'])
    elif args.IC['type'] == 'sinusoidal':
        u0, ut0, u0_hat, ut0_hat, x = wave_sample_initial_conditions_sinusoidal(x_vals, L)
    
    # Convert numpy arrays to torch tensors
    u0_hat = torch.from_numpy(u0_hat)
    ut0_hat = torch.from_numpy(ut0_hat)
    
    # Define spectral RHS function
    def spectral_rhs(u_hat, ut_hat, k_vals, c):
        return - (c**2) * (k_vals**2) * u_hat  # Second derivative in Fourier space

    # Initialize storage
    u_fine = torch.zeros((len(t_fine), Nx), dtype=torch.cfloat)
    ut_fine = torch.zeros((len(t_fine), Nx), dtype=torch.cfloat)
    energy_vals = []
    
    # Initialize Fourier coefficients
    u_hat = u0_hat.clone()
    ut_hat = ut0_hat.clone()
    
    # Compute initial energy
    initial_energy = torch.sum(torch.abs(ut_hat)**2 + c**2 * torch.abs(k * u_hat)**2) * (L / Nx)
    energy_vals.append(initial_energy)

    # Time-stepping using RK4
    for i in range(len(t_fine)):
        u_fine[i, :] = torch.fft.ifft(u_hat).real
        ut_fine[i, :] = torch.fft.ifft(ut_hat).real

        # Compute Runge-Kutta 4 (RK4) stages
        k1 = dt * ut_hat
        l1 = dt * spectral_rhs(u_hat, ut_hat, k, c)
        
        k2 = dt * (ut_hat + 0.5 * l1)
        l2 = dt * spectral_rhs(u_hat + 0.5 * k1, ut_hat + 0.5 * l1, k, c)
        
        k3 = dt * (ut_hat + 0.5 * l2)
        l3 = dt * spectral_rhs(u_hat + 0.5 * k2, ut_hat + 0.5 * l2, k, c)
        
        k4 = dt * (ut_hat + l3)
        l4 = dt * spectral_rhs(u_hat + k3, ut_hat + l3, k, c)
        
        # Update values
        u_hat += (k1 + 2*k2 + 2*k3 + k4) / 6
        ut_hat += (l1 + 2*l2 + 2*l3 + l4) / 6

        # Compute energy at this time step
        energy_t = torch.sum(torch.abs(ut_hat)**2 + c**2 * torch.abs(k * u_hat)**2) * (L / Nx)
        energy_vals.append(energy_t)
    
    # Check energy conservation
    energy_vals = torch.tensor(energy_vals)
    avg_energy = torch.mean(energy_vals)
    max_energy_dev = torch.max(torch.abs(energy_vals - initial_energy))
    assert max_energy_dev < 1, f"Energy not conserved! Max deviation: {max_energy_dev}"

    # Extract values at original timepoints
    t_indices = torch.searchsorted(t_fine, torch.tensor(t_vals))
    u_data = u_fine[t_indices]
    ut_data = ut_fine[t_indices]
    energy_vals = energy_vals[t_indices]
    
    return x, t_vals, torch.tensor(u_data.real), torch.tensor(ut_data.real), energy_vals.numpy(), torch.tensor(u0), torch.tensor(ut0)

# def wave_evolve_fft(args, x_vals, t_vals, c):
#     """Evolve the wave equation using FFT-based spectral solution."""
#     Nx = len(x_vals)
#     Nt = len(t_vals)    
#     L = x_vals[-1] - x_vals[0]  # Domain length

#     k = np.fft.fftfreq(Nx, d=(L/Nx)) * 2 * np.pi  # Wave numbers

#     # Sample initial conditions
#     if args.IC['type'] == 'periodic_gp':
#         u0, ut0, u0_hat, ut0_hat, x = wave_sample_initial_conditions_periodic_gp(x_vals, L, **args.IC['params'])
#     elif args.IC['type'] == 'sinusoidal':
#         u0, ut0, u0_hat, ut0_hat, x = wave_sample_initial_conditions_sinusoidal(x_vals, L)

#     # Compute Fourier coefficients
#     A_n = u0_hat
#     B_n = np.zeros_like(A_n)
    
#     nonzero_indices = np.abs(k) > 1e-10  
#     B_n[nonzero_indices] = ut0_hat[nonzero_indices] / (c * k[nonzero_indices])
#     B_n[0] = ut0_hat[0] / c  # Handle k=0 case correctly

#     # Initialize storage
#     energy_vals = []
#     u_data = np.zeros((Nt, Nx), dtype=complex)
#     ut_data = np.zeros((Nt, Nx), dtype=complex)

#     initial_energy = np.sum(np.abs(ut0_hat)**2 + c**2 * np.abs(k * u0_hat)**2) * L / (Nx ** 2)
#     print(initial_energy)

#     for i, t in enumerate(t_vals):
#         cos_term = np.cos(c * k * t)
#         sin_term = np.sin(c * k * t)
        
#         # Compute u(x,t) and u_t(x,t) in Fourier space
#         u_hat_t = A_n * cos_term + B_n * sin_term
#         ut_hat_t = -c * k * A_n * sin_term + c * k * B_n * cos_term

#         # Transform back to physical space
#         u_ifft = np.fft.ifft(u_hat_t)
#         ut_ifft = np.fft.ifft(ut_hat_t)
        
#         # Check imaginary part
#         assert np.max(np.abs(u_ifft.imag)) < 1e-5, f"Warning: Nonzero imaginary part detected in u! {np.max(np.abs(u_ifft.imag))} at iteration {i}"
#         assert np.max(np.abs(ut_ifft.imag)) < 1e-5, f"Warning: Nonzero imaginary part detected in ut! {np.max(np.abs(ut_ifft.imag))} at iteration {i}"

#         u_data[i, :] = u_ifft.real
#         ut_data[i, :] = ut_ifft.real

#         # Compute energy
#         energy_t = np.sum(np.abs(ut_hat_t)**2 + c**2 * np.abs(k * u_hat_t)**2) * L / (Nx ** 2)
#         energy_vals.append(energy_t)
        
#     print(energy_vals)

#     return x, t_vals, u_data, ut_data, energy_vals, u0, ut0

def get_1d_wave_data(args):
    c = args.IC['c']
    t_vals = np.linspace(args.tmin, args.tmax, args.Nt, endpoint=False)
    x_vals = np.linspace(args.xmin, args.xmax, args.Nx, endpoint=False)

    Nx = args.Nx
    Nt = args.Nt

    if args.method == 'deeponet' or args.method == 'orthonormal_pushforward':
        branch_data = None
        trunk_t = None
        y = None
        L = args.xmax - args.xmin
        for i in range(args.n_branch):
            x, t_vals, u_data, ut_data, energy_values, u0, ut0 = wave_evolve_fft(args, x_vals, t_vals, c)
            # print(f'energy values are {energy_values}')

            k = torch.tensor(np.fft.fftfreq(args.Nx, d=((args.xmax-args.xmin)/args.Nx)) * 2* np.pi).float()  # to device?

            branch_block = np.stack([u0,ut0], axis=0)
            branch_block = np.stack([branch_block] * t_vals.shape[0], axis=0)

            gt_u = torch.tensor(u0[None,:])
            gt_ut = torch.tensor(ut0[None,:])
            gt_u_hat = torch.fft.fft(gt_u, dim=1)  # dim=1 since shape is (time, space)
            gt_ut_hat = torch.fft.fft(gt_ut, dim=1)
            init_energy = torch.sum(torch.abs(gt_ut_hat)**2 + args.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) * L / (args.Nx ** 2)


            y_block = np.stack([u_data,ut_data], axis=0)
            y_block = np.stack([u_data, ut_data], axis=0).transpose(1, 0, 2)
            y_block = y_block.reshape(-1, 2, args.Nx)  # reshape to (time, 2, space)

            gt_u = torch.tensor(u_data)
            gt_ut = torch.tensor(ut_data)

            gt_u_hat = torch.fft.fft(gt_u, dim=1)  # dim=1 since shape is (time, space)
            gt_ut_hat = torch.fft.fft(gt_ut, dim=1)
            target_energy = torch.sum(torch.abs(gt_ut_hat)**2 + args.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) * L / (args.Nx ** 2)

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

        gt_u_hat = torch.fft.fft(gt_u, n=args.Nx, dim=1)  # dim=1 since shape is (time, space)
        gt_ut_hat = torch.fft.fft(gt_ut, n=args.Nx, dim=1)
        true_energy = torch.sum(torch.abs(gt_ut_hat)**2 + args.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) / args.Nx

        #target energy
        gt_u = torch.tensor(y[:,0,:])
        gt_ut = torch.tensor(y[:,1,:])

        gt_u_hat = torch.fft.fft(gt_u, n=args.Nx, dim=1)  # dim=1 since shape is (time, space)
        gt_ut_hat = torch.fft.fft(gt_ut, n=args.Nx, dim=1)
        target_energy = torch.sum(torch.abs(gt_ut_hat)**2 + args.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) / args.Nx

        assert (true_energy - target_energy).max() < 1, f'Energy is not conserved in training data – max difference was: {(true_energy - target_energy).max()}'

        data = DeepOData(x, y)
    elif args.method == 'full_fourier':
        x = None
        y = None
        for i in range(args.n_branch):
            x_vals, t_vals, u_data, ut_data, energy_values, u0, ut0 = wave_evolve_fft(args, x_vals, t_vals, c)
            x_block = torch.concatenate((u0.unsqueeze(0), ut0.unsqueeze(0)), axis=0).unsqueeze(0) #n_branch x 2 x num_x
            y_block = torch.concatenate((u_data.unsqueeze(0),ut_data.unsqueeze(0)), axis=0).unsqueeze(0) #n_branch x 2 x num_x 
            # y_block = y_block.reshape(-1, 1).reshape(1,2,-1) 
            if x is None:
                x = x_block
                y = y_block
            else:
                x = torch.concatenate((x, x_block), axis=0)
                y = torch.concatenate((y, y_block), axis=0)
        
        #transpose last two dimensions
        y = y.transpose(2,3)

        print(y.shape)



        data = SpectralSpaceTime(x, y)
    else:
        raise ValueError(f"Method {args.method} not implemented as data type.")

    return data
