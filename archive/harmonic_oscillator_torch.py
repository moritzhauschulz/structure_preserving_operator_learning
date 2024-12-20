import numpy as np
from scipy import io
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os
from itertools import cycle
from utils.model import DeepONet
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from deepxde.optimizers.pytorch.optimizers import _get_learningrate_scheduler 


np.random.seed(42)

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
        self.labels = torch.tensor(y)

    def __len__(self):
        return len(self.branch_data)  # Total size of the dataset

    def __getitem__(self, idx):
        return (self.branch_data[idx], self.trunk_data[idx]), self.labels[idx]  # Return a single data-label pair


def print_summary(data):
    # Compute summary statistics by columns
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    # Print the summary statistics
    print("Summary Statistics by Column:")
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
    print(f"Minimum: {min_val}")
    print(f"Maximum: {max_val}")

# Make dataset
def get_data(load=False, save=True, n_branch=50, n_trunk=1000, q0=None, p0=None, tmin=0, tmax=25):
    if load and os.path.exists('harmonic_oscillator_datasets.pkl'):
        with open('harmonic_oscillator_datasets.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        if q0 is None:
            q0 = np.random.uniform(-1, 1, size=(n_branch, 1))  # Initial position (x(0))
        else: 
            q0 = np.ones((n_branch,1)) * q0
        if p0 is None: 
            p0 = np.random.uniform(-1, 1, size=(n_branch, 1))  # Initial velocity (dx/dt at t=0)
        else:
            p0 = np.ones((n_branch,1)) * p0
        omega = np.random.uniform(0.5, 1, size=(n_branch, 1)) # Angular frequency
        trunk_data =  np.linspace(0, tmax, n_trunk)

        branch_data = np.concatenate((q0,p0,omega), axis=1) #n_branch x 3
        # print_summary(branch_data)
        t_span = (0, tmax)

        # print(trunk_data.shape)
        # print(branch_data.shape)

        #generate y
        y = np.array([])
        for i in range(n_branch):
            # Initial conditions
            z0 = branch_data[i,0:2] #initial phase space
            (omega) = branch_data[i,2] #angular velocity
            # Solve the ODE
            solution = solve_ivp(harmonic_oscillator, t_span, z0, args=(omega,), t_eval=trunk_data)
            # print(solution.y[0])
            # Extract the solution
 
            y = np.concatenate([y,np.array(solution.y[0])])


         

        trunk_data = np.tile(trunk_data, (1, n_branch)).transpose() #n_trunk * n_trunk
        branch_data = np.repeat(branch_data, repeats=n_trunk, axis=0)

        x = (branch_data.astype(np.float32), trunk_data.astype(np.float32))
        y = np.expand_dims(y, axis=1).astype(np.float32)

        # plot_y_with_initial_conditions(x, y, n_branch=n_branch, n_trunk=n_trunk)

        # print(x[0].shape)
        # print(x[1].shape)
        # print(y.shape)

        data = DeepOData(x, y)
        if save:
            with open('harmonic_oscillator_datasets.pkl', 'wb') as f:
                pickle.dump(data, f)

    return data

def train(model, lr, epochs):
    decay = ("inverse time", epochs // 5, 0.5)
    model.compile("adam", lr=lr, metrics=["mean l2 relative error"], decay=decay)
    losshistory, train_state = model.train(epochs=epochs, batch_size=None)
    visualize_loss(losshistory)
    print("\nTraining done ...\n")

def main():
    tmax = 25
    lr = 0.001
    epochs = 1000 
    data = get_data(tmax=tmax)
    print(data.branch_data[0])

    train_loader = DataLoader(data, batch_size=1024, shuffle=True)

    model = DeepONet(activation=nn.Tanh)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    decay = ("inverse time", epochs // 5, 0.5)
    scheduler = _get_learningrate_scheduler(optimizer, decay)

    compute_loss = torch.nn.MSELoss()

    losses = []
    
    pbar = tqdm(range(epochs))
    for i in pbar:
        for batch_idx, (x, y) in enumerate(train_loader):
            
            preds = model(x)

            # print(f'preds shape {preds.shape}')
            # print(f'preds shape {y.shape}')

            loss = compute_loss(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print_summary(np.array(x[0]))

        losses.append(loss.item())
        pbar.set_description(f'epoch {i}; loss {loss.item()}')

        scheduler.step()

    print('Training completed.')
    print('Generating examples...')
    
    visualize_loss(losses)

    tmax_pred = 50
    examples = [data.branch_data[0]]#torch.tensor(([0.0,0.5,0.5],[0.5,0.5,0.5],[-0.5,0.5,0.5],[1,0.5,0.5],[-1,0.5,0.5]))
    example_t = torch.linspace(0,50,500).unsqueeze(0).T
    output = []
    ground_truth = []
    for x in examples:
        example_u = x.unsqueeze(0).T
        output.append(torch.squeeze(model((example_u.T, example_t))).detach())
        ground_truth.append(solve_ivp(harmonic_oscillator, (0, tmax_pred), np.squeeze(example_u[0:2]), args=(example_u[2].item(),), t_eval=np.squeeze(example_t)).y[0])
    visualize_example(examples, example_t, output, ground_truth)

def visualize_loss(losses):
    train_loss = losses
    steps = range(len(losses))

    # Plotting the curves
    plt.figure(figsize=(8, 6))
    plt.plot(steps, train_loss, label='Train Loss', marker='o')
    # plt.plot(steps, test_loss, label='Test Loss', marker='s')
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss Curves', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')  # Save as a PNG file
    # plt.show()

    return None

def visualize_example(labels,x,y_hat,y):
    # Plotting the curves
    plt.figure(figsize=(8, 6))
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])  # or define your own colors

    for i, label in enumerate(labels):
        color = next(colors)
        plt.plot(x, y[i], label=f'{label} – ground truth', linestyle='--', color=color)
        plt.plot(x, y_hat[i], label=f'{label} – prediction', linestyle='solid', color=color)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('time', fontsize=14)
    plt.title('Prediction vs Ground Truth', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')  # Save as a PNG file
    # plt.show()

    return None

def plot_y_with_initial_conditions(x, y, n_branch, n_trunk):
    """
    Generate a plot for the solution `y` of the harmonic oscillator and include values of z0 and omega.

    Parameters:
        y (np.ndarray): Array containing the solution values (position y(t)).
        z0 (list or np.ndarray): Initial conditions [position, velocity].
        omega (float): Angular velocity.
        trunk_data (np.ndarray): Time points corresponding to the solution `y`.
    """
    # Plot the results
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])  # or define your own colors

    plt.figure(figsize=(8, 5))
    for i in range(n_branch):
        i *= n_trunk
        print(f'i is {i}')
        color =next(colors)
        print(color)
        for j in range(i, i+n_trunk-1):
            if sum(x[0][j,0:2]) != sum(x[0][j+1,0:2]):
                print(f'non-uniform input at j={j} and neighbour')
        plt.plot(x[1][i:i+n_trunk], x[0][i:i+n_trunk,0], label=f"q0", color=color, linestyle='--')
        plt.plot(x[1][i:i+n_trunk], x[0][i:i+n_trunk,1]*x[1][i:i+n_trunk] + x[0][i:i+n_trunk,0], label=f"p0", color=color, linestyle=':')
        plt.plot(x[1][i:i+n_trunk], y[i:i+n_trunk], label=f"Solution y(t)", color=color)
    # plt.title(f"Harmonic Oscillator Solution\nInitial Conditions: z0={z0}, \u03c9={omega}")
    plt.xlabel("Time t")
    plt.ylabel("Position y(t)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()







