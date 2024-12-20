import matplotlib.pyplot as plt
from itertools import cycle
from typing import List, Union
import numpy as np
import torch


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
    plt.savefig(args.save_dir + '/predictions.png', dpi=300, bbox_inches='tight')


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
    plt.grid(True)
    plt.show()

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
    plt.savefig(args.save_dir + '/loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def compute_example_with_energy(args, val_data, model):
    examples = [val_data.branch_data[0]]
    ground_truth = [val_data.labels[0:args.n_trunk]]
    example_t = val_data.trunk_data[0:args.n_trunk]
    for x in examples:
        example_u = x.unsqueeze(0).T
        output = model((example_u.T, example_t.requires_grad_(True)))
        gradients = torch.autograd.grad(outputs=output[:, 0], inputs=example_t, grad_outputs=torch.ones_like(output[:, 0]), create_graph=True)[0]
        output = output.detach()
        gradients = torch.squeeze(gradients[0]).detach()
        nrg_hat = 0.5 * output[:,0] **2 + 0.5 * gradients**2
        vel_nrg_hat = 0.5 * output[:,0] **2 + 0.5 * output[:,1]**2
        nrg = (0.5 * (example_u[0] * example_u[1]) ** 2 + 0.5 * example_u[2] ** 2).repeat(len(nrg_hat))
    example_t = example_t.detach()
    return examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, nrg

def visualize_example_with_energy(args, labels, y, out, out_hat, nrg, nrg_hat, vel_nrg_hat):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    ax2 = ax1.twinx()

    lines = []
    legend_labels = []

    for i, label in enumerate(labels):
        color = next(colors)
        line1, = ax1.plot(y, out[i][:, 0], label=f'{label} – position ground truth', linestyle='--', color=color)
        line2, = ax1.plot(y, out_hat[:, 0], label=f'{label} – position prediction', linestyle='solid', color=color)
        line3, = ax1.plot(y, out[i][:, 1], label=f'{label} – velocity ground truth', linestyle='--', color=next(colors))
        line4, = ax1.plot(y, out_hat[:, 1], label=f'{label} – velocity prediction', linestyle='solid', color=next(colors))
        line5, = ax2.plot(y, nrg, label=f'{label} – init energy', linestyle=':', color=color)
        line6, = ax2.plot(y, nrg_hat, label=f'{label} – gradient-predicted energy', linestyle='-.', color=color)
        line7, = ax2.plot(y, vel_nrg_hat, label=f'{label} – velocity-predicted energy', linestyle='-.', color=color)
        lines.extend([line1, line2, line3, line4, line5, line6, line7])
        legend_labels.extend([line1.get_label(), line2.get_label(), line3.get_label(), line4.get_label(), line5.get_label(), line6.get_label(), line7.get_label()])

    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Position / Velocity', fontsize=14)
    ax2.set_ylabel('Energy', fontsize=14)
    ax1.set_title('Prediction vs Ground Truth with Energy', fontsize=16)
    ax1.grid(True)
    fig.tight_layout()
    ax1.legend(lines, legend_labels, fontsize=12, loc='upper left')
    plt.savefig(args.save_dir + '/predictions_with_energy.png', dpi=300, bbox_inches='tight')
    plt.show()