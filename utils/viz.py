import matplotlib.pyplot as plt
from itertools import cycle
import torch
import wandb
import pandas as pd
from copy import deepcopy
import numpy as np
from .data import exact_soliton

def visualize_example(args,labels, x, y_hat, y):
    plt.figure(figsize=(8, 6))
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for i, label in enumerate(labels):
        color = next(colors)
        plt.plot(x, y[i], label=f'{label} – ground truth', linestyle='--', color=color)
        plt.plot(x, y_hat[i], label=f'{label} – prediction', linestyle='solid', color=color)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('time', fontsize=14)
    plt.title('Prediction vs Ground Truth', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    #expand plot slightly
    y_min, y_max = plt.ylim()
    padding = 0.1 * (y_max - y_min)  # Add 10% padding
    plt.ylim(y_min - padding, y_max + padding)

    plt.savefig(args.save_dir + '/predictions.png', dpi=300, bbox_inches='tight')


def plot_y_with_initial_conditions(x, y, n_branch, n_trunk):
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    plt.figure(figsize=(8, 5))
    for i in range(n_branch):
        i *= n_trunk
        color = next(colors)
        for j in range(i, i + n_trunk - 1):
            if sum(x[0][j, 0:2]) != sum(x[0][j + 1, 0:2]):
                print(f'non-uniform input at j={j} and neighbour')
        plt.plot(x[1][i:i + n_trunk], x[0][i:i + n_trunk, 0], label=f"q0", color=color, linestyle='--')
        plt.plot(x[1][i:i + n_trunk], x[0][i:i + n_trunk, 1] * x[1][i:i + n_trunk] + x[0][i:i + n_trunk, 0], label=f"p0", color=color, linestyle=':')
        plt.plot(x[1][i:i + n_trunk], y[i:i + n_trunk], label=f"Solution y(t)", color=color)
    plt.xlabel("Time t")
    plt.ylabel("Position y(t)")
    plt.legend()

    #expand plot slightly
    y_min, y_max = plt.ylim()
    padding = 0.1 * (y_max - y_min)  # Add 10% padding
    plt.ylim(y_min - padding, y_max + padding)

    plt.grid(True)

def visualize_loss(args, train_losses, val_losses):
    steps = range(len(train_losses))

    plt.figure(figsize=(8, 6))
    plt.plot(steps, train_losses, label='Train Loss', marker='o')
    plt.plot(steps, val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss Curves', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(args.save_plots + '/loss_curves.png', dpi=300, bbox_inches='tight')

def compute_example_with_energy(i, args, data, model):
    ground_truth = data.labels[i*args.n_trunk:(i+1)*args.n_trunk]
    example_t = data.trunk_data[i*args.n_trunk:(i+1)*args.n_trunk].to(args.device)
    example_u = data.branch_data[i*args.n_trunk].unsqueeze(0).T.to(args.device)
    output = model((example_u.T.repeat(args.n_trunk, 1), example_t.requires_grad_(True)))
    if isinstance(output, tuple):
        output = output[0]
    gradients = torch.autograd.grad(outputs=output[:, 0], inputs=example_t, grad_outputs=torch.ones_like(output[:, 0]), create_graph=True)[0]
    output = output.detach().cpu()
    example_u = example_u.detach().cpu()
    example_t = example_t.detach().cpu()
    gradients = torch.squeeze(gradients).detach().cpu()
    nrg_hat = 0.5 * (example_u[2] * output[:,0]) **2 + 0.5 * gradients**2
    if args.num_outputs == 2:
        vel_nrg_hat = 0.5 * (example_u[2] * output[:,0]) **2 + 0.5 * (output[:,1]) **2
        numerical_nrg = 0.5 * (example_u[2].repeat(args.n_trunk) * ground_truth[:,0]) ** 2 + 0.5 * (ground_truth[:,1]) ** 2
    else:
        vel_nrg_hat = None
        numerical_nrg = None
    nrg = (0.5 * (example_u[2] * example_u[0]) ** 2 + 0.5 * example_u[1] ** 2).repeat(args.n_trunk)
    example_t = example_t.detach()
    return (
        example_u,
        example_t,
        ground_truth,
        output,
        nrg_hat,
        vel_nrg_hat,
        numerical_nrg,
        nrg,
        gradients
    )

def visualize_example_with_energy(example_type, args, label, y, out, out_hat, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, gradients):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    ax2 = ax1.twinx()
    ax2.set_ylim(bottom=0)  # Ensure ax2 starts at 0

    lines = []
    legend_labels = []

    color = next(colors)
    line1, = ax1.plot(y, out[:, 0], label=f'position ground truth', linestyle='--', color=color)
    line2, = ax1.plot(y, out_hat[:, 0], label=f'position prediction', linestyle='solid', color=color)
    color = next(colors)
    line3, = ax2.plot(y, nrg, label=f'init energy', linestyle='-', color=color)
    color = next(colors)
    #line4, = ax2.plot(y, nrg_hat, label=f'gradient-predicted energy', linestyle='-.', color=color)
    #color = next(colors)
    #line5, = ax1.plot(y, gradients, label=f'gradient', linestyle=':', color=color)
    lines.extend([line1, line2, line3]) #, line4, line5
    legend_labels.extend([line1.get_label(), line2.get_label(), line3.get_label()]) # line4.get_label(), line5.get_label()
    if args.num_outputs == 2:
        line6, = ax1.plot(y, out[:, 1], label=f'velocity ground truth', linestyle='--', color=color)
        line7, = ax1.plot(y, out_hat[:, 1], label=f'velocity prediction', linestyle='solid', color=color)
        color = next(colors)
        line8, = ax2.plot(y, vel_nrg_hat, label=f'velocity-predicted energy', linestyle=(0,(1, 1)), color=color)
        color = next(colors)
        # line9, = ax2.plot(y, numerical_nrg, label=f'numerically-predicted energy', linestyle='', color=color)  # numerical energy
        lines.extend([line6, line7, line8, line8])
        legend_labels.extend([line6.get_label(), line7.get_label(), line8.get_label()])
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Position / Velocity', fontsize=14)
    ax2.set_ylabel('Energy', fontsize=14)
    ax1.set_title(f'Prediction vs Ground Truth with Energy – q0={label[0].item():.2f}, p0={label[1].item():.2f}, omega={label[2].item():.2f}', fontsize=16)
    ax1.grid(True)
    fig.tight_layout()
    ax1.legend(lines, legend_labels, fontsize=12, loc='upper right')

    #expand plot slightly
    y_min, y_max = plt.ylim()
    padding = 0.1 * (y_max - y_min)  # Add 10% padding

    plt.ylim(y_min - padding, y_max + padding)
    plt.savefig(args.save_plots + f'/predictions_with_energy_q0={label[0].item():.2f}_p0={label[1].item():.2f}_omega={label[2].item():.2f}_{example_type}.png', dpi=300, bbox_inches='tight')

def wandb_viz_loss(exp_n, save_dir=None, exclude_val=False, wandb_user='moritz-hauschulz', wandb_project='structure-preserving-operator-learning', losses=None, vars=None):
    
    #get the wandb run data imported
    api = wandb.Api()
    run = api.run(f"{wandb_user}/{wandb_project}/{exp_n}")

    metrics = run.history()  # Replace "metric_name" with the logged key

    if losses == None:
        losses = []
        for metric in metrics.keys():
            if 'loss' in metric:
                losses.append(metric)
        print(f'losses')
    
    plt.figure(figsize=(8, 6))

    markers = cycle(['o', 'x', '*', '+'])
    colours_train = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colours_val = deepcopy(colours_train)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = None  # For secondary y-axis, if needed

    for loss in sorted(losses):
        if 'val' in loss:
            style = '--'
            color = next(colours_val)
        else:
            style = '-'
            color = next(colours_train)

        iteration = metrics['epoch']
        loss_data = metrics[loss]

        valid_data = [(i, l) for i, l in zip(iteration, loss_data) if not pd.isna(l)]

        # Separate into x (iterations) and y (loss values)
        if valid_data:
            x, y = zip(*valid_data)
        else:
            continue

        # Check if this loss should go on a secondary y-axis
        if vars and loss in vars:
            if ax2 is None:  # Create the secondary axis if it doesn't exist
                ax2 = ax1.twinx()
                ax2.set_ylabel("Loss", fontsize=14)
            ax2.plot(x, y, label=loss, linestyle=style, color=color)
        else:
            ax1.plot(x, y, label=loss, linestyle=style, color=color)

    # Customize the axes
    ax1.set_xlabel("Iteration", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.set_title("Loss Curves", fontsize=16)

    # Create legends
    ax1_legend = ax1.legend(fontsize=10, loc='upper right', title="Left Axis")
    ax1.add_artist(ax1_legend)
    if ax2:
        ax2.legend(fontsize=10, loc='center right', title="Right Axis")

    # Save or show the plot
    if save_dir:
        plt.savefig(f"{save_dir}/loss_curves.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_1d_KdV_Soliton(args, h, x_res, a, c, model, save_dir):
    L = args.xmax - args.xmin

    print(args.xmax, args.xmin, L)

    t_values = np.arange(args.tmin, args.tmax + h, h)  # Time steps for saving solutions
    x = np.linspace(-L/2, L/2, int(L/x_res), endpoint=False)

    def exact_soliton(x, t, c, a):
        arg = np.clip(np.sqrt(c) * (x - c * t - a) / 2, -50, 50)  # Prevent extreme values
        return (c / 2) / np.cosh(arg) ** 2  # Stable sech^2 computation

    plt.figure(figsize=(10, 6))

    for t in t_values:
        t_tensor = torch.tensor(t).repeat(x.shape[0], 1)
        tx = torch.cat([t_tensor, torch.tensor(x).unsqueeze(-1)],dim=1).float().to(args.device)
        IC = torch.tensor([a, c]).float().unsqueeze(0).repeat(x.shape[0], 1).to(args.device)
        out = model((IC, tx))
        y_hat = out[0].detach().cpu().numpy()
        fourier_nrg = out[1][-1]
        y = exact_soliton(x, t, c, a)

        #compute energy of y via fourier modes

        y_cumulative = np.cumsum(y**2) * x_res
        energy_y = np.sum(y**2) * (x_res)  # L2 norm squared times dx
        y_hat_cumulative = np.cumsum(y_hat**2) * x_res
        energy_y_hat = np.sum(y_hat**2) * (x_res)
        print(f'min and max fourier nrg: {fourier_nrg.min().item()}, {fourier_nrg.max().item()}')
        print(f"Energy at t={t:.1f}: True = {energy_y:.6f}, Predicted = {energy_y_hat:.6f}")

        plt.plot(x, y, label=f"t = {t:.1f}, nrg={energy_y:.2f}", linestyle='-')
        plt.plot(x, y_hat, label=f"t = {t:.1f} (predicted), nrg={energy_y_hat:.2f}", linestyle='--')
        plt.plot(x, y_cumulative, label=f"t = {t:.1f} (cumulative)", linestyle='-.')
        plt.plot(x, y_hat_cumulative, label=f"t = {t:.1f} (predicted cumulative)", linestyle=':')

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.title("KdV Equation: Exact vs Predicted Evolution")
    plt.grid()
    if save_dir:
        plt.savefig(f"{save_dir}/KdV_preds.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_1d_KdV_Soliton_ifft(args, h, x_res, a, c, model, save_dir):
    L = args.xmax - args.xmin

    print(args.xmax, args.xmin, L)

    t_values = np.arange(args.tmin, args.tmax + h, h)  # Time steps for saving solutions
    x = np.linspace(-L/2, L/2, args.col_N)

    plt.figure(figsize=(10, 6))

    for t in t_values:
        t_tensor = torch.tensor(t).unsqueeze(-1).unsqueeze(-1).float().to(args.device)
        IC = torch.tensor([a, c]).float().unsqueeze(0).to(args.device)
        out = model((IC, t_tensor))
        y_hat = out[0].T.squeeze(-1).detach().cpu().numpy()
        fourier_nrg = out[1][-1]
        y = exact_soliton(x, t, c, a)

        mse_loss = torch.nn.MSELoss()

        print(f'mse loss is: {mse_loss(torch.tensor(y),torch.tensor(y_hat))}')

        #compute energy of y via fourier modes

        # y_cumulative = np.cumsum(y**2) * x_res
        energy_y = np.sum(y**2) * (x[1]-x[0])  # L2 norm squared times dx
        # y_hat_cumulative = np.cumsum(y_hat**2) * x_res
        energy_y_hat = np.sum(y_hat**2) * (x[1]-x[0])
        print(f'min and max fourier nrg: {fourier_nrg.min().item()}, {fourier_nrg.max().item()}')
        print(f"Energy at t={t:.1f}: True = {energy_y:.6f}, Predicted = {energy_y_hat:.6f}")
        
        print(x.shape, y_hat.shape)
        plt.plot(x, y, label=f"t = {t:.1f}, nrg={energy_y:.2f}", linestyle='-')
        plt.plot(x, y_hat, label=f"t = {t:.1f} (predicted), nrg={energy_y_hat:.2f}", linestyle='--')
        # plt.plot(x, y_cumulative, label=f"t = {t:.1f} (cumulative)", linestyle='-.')
        # plt.plot(x, y_hat_cumulative, label=f"t = {t:.1f} (predicted cumulative)", linestyle=':')

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.title("KdV Equation: Exact vs Predicted Evolution")
    plt.grid()
    if save_dir:
        plt.savefig(f"{save_dir}/KdV_preds.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


    