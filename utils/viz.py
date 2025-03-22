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
    dummy = None
    ground_truth = data.labels[i*args.n_trunk:(i+1)*args.n_trunk]
    example_t = data.trunk_data[i*args.n_trunk:(i+1)*args.n_trunk].to(args.device)
    example_u = data.branch_data[i*args.n_trunk].unsqueeze(0).T.to(args.device)
    output = model((example_u.T.repeat(args.n_trunk, 1), example_t.requires_grad_(True)), dummy, dummy)
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

    dummy = None
    fig, ax1 = plt.subplots(figsize=(8, 6))
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    ax2 = ax1.twinx()
    ax2.set_ylim(bottom=0)  # Ensure ax2 starts at 0

    lines = []
    legend_labels = []

    color = next(colors)
    line1, = ax1.plot(y, out[:, 0], label=f'true position', linestyle='--', color=color)
    line2, = ax1.plot(y, out_hat[:, 0], label=f'predicted position', linestyle='solid', color=color)
    color = next(colors)
    line3, = ax2.plot(y, nrg, label=f'true energy', linestyle='-', color=color)
    color = next(colors)
    #line4, = ax2.plot(y, nrg_hat, label=f'gradient-predicted energy', linestyle='-.', color=color)
    #color = next(colors)
    #line5, = ax1.plot(y, gradients, label=f'gradient', linestyle=':', color=color)
    lines.extend([line1, line2, line3]) #, line4, line5
    legend_labels.extend([line1.get_label(), line2.get_label(), line3.get_label()]) # line4.get_label(), line5.get_label()
    if args.num_outputs == 2:
        line6, = ax1.plot(y, out[:, 1], label=f'true velocity', linestyle='--', color=color)
        line7, = ax1.plot(y, out_hat[:, 1], label=f'predicted velocity', linestyle='solid', color=color)
        color = next(colors)
        line8, = ax2.plot(y, vel_nrg_hat, label=f'velocity-predicted energy', linestyle=(0,(1, 1)), color=color)
        color = next(colors)
        # line9, = ax2.plot(y, numerical_nrg, label=f'numerically-predicted energy', linestyle='', color=color)  # numerical energy
        lines.extend([line6, line7, line8, line8])
        legend_labels.extend([line6.get_label(), line7.get_label(), line8.get_label()])
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Position / Velocity', fontsize=14)
    ax2.set_ylabel('Energy', fontsize=14)
    ax1.set_title(f'Prediction vs Ground Truth with Energy – $q_0$={label[0].item():.2f}, $p_0$={label[1].item():.2f}, $\\omega$={label[2].item():.2f}', fontsize=16)
    ax1.grid(True)
    fig.tight_layout()
    ax1.legend(lines, legend_labels, fontsize=12, bbox_to_anchor=(1.15, 1), loc='upper left')

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

    h=1

    t_values = np.arange(args.tmin, args.tmax + h, h)  # Time steps for saving solutions
    x = np.linspace(-L/2, L/2, args.Nx)

    plt.figure(figsize=(10, 6))

    # Use a colorblind-friendly palette
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    color_cycle = cycle(colors)
    
    for t in t_values:
        t_tensor = torch.tensor(t).unsqueeze(-1).unsqueeze(-1).float().to(args.device)
        IC = torch.tensor([a, c]).float().unsqueeze(0).to(args.device)
        out = model((IC, t_tensor))
        y_hat = out[0].T.squeeze(-1).detach().cpu().numpy()
        fourier_nrg = out[1][-1]
        y = exact_soliton(x, t, c, a)

        mse_loss = torch.nn.MSELoss()
        print(f'mse loss is: {mse_loss(torch.tensor(y),torch.tensor(y_hat))}')

        energy_y = np.sum(y**2) * (x[1]-x[0])
        energy_y_hat = np.sum(y_hat**2) * (x[1]-x[0])
        print(f"Energy at t={t:.1f}: True = {energy_y:.6f}, Predicted = {energy_y_hat:.6f}")
        print(f'min and max fourier nrg: {fourier_nrg.min().item()}, {fourier_nrg.max().item()}')

        color = next(color_cycle)
        plt.plot(x, y, label=f"t = {t:.1f}, nrg={energy_y:.3f}", 
                linestyle='-', color=color)
        plt.plot(x, y_hat, label=f"t = {t:.1f}, nrg={fourier_nrg.mean():.3f} (predicted)", 
                linestyle='--', color=color)

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.title("KdV Equation: Exact vs Predicted Evolution")
    plt.grid()
    if save_dir:
        plt.savefig(f"{save_dir}/KdV_preds.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_1d_wave_IC(args, x, u0, ut0, save_dir):
    # Plot Initial Conditions
    plt.figure(figsize=(10, 4))
    plt.plot(x, u0, label=r'$u(0, x)$', linewidth=2)
    plt.plot(x, ut0, label=r'$\partial_t u(0, x)$', linestyle='dashed', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.title('Initial Conditions')
    plt.legend()
    plt.grid()
    if save_dir:
        plt.savefig(f"{save_dir}/1d_wave_IC.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_1d_wave_energy(args, t_vals, energy_values, save_dir):
    # Plot energy conservation
    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, energy_values, label=r'Energy via Parseval’s theorem', color='red')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Exact Energy Conservation of the Wave Equation')
    plt.legend()
    plt.grid()
    plt.show()
    if save_dir:
        plt.savefig(f"{save_dir}/1d_wave_IC_nrg.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

# Plot kdv evolution
def plot_1d_KdV_evolution(args, i, data, model, save_dir=None, val=False) :

    if val:
        suffix = '_val'
    else:    
        suffix = '' 
    
    num_t = int((args.tmax - args.tmin)/(args.t_res))

    gt_u = data.labels[i*num_t:(i+1)*num_t]
    example_t = data.trunk_data[i*num_t:(i+1)*num_t].to(args.device)
    example_u = data.branch_data[i*num_t].view(1,-1).to(args.device)

    x=(data.branch_data[i*num_t].unsqueeze(0).repeat(num_t,1,1),)
    y=data.labels[i*num_t:(i+1)*num_t]

    all_output = model((example_u.repeat(num_t, 1), example_t.requires_grad_(True)),x=x,y=y)
    output = all_output[0]
    true_energy = all_output[1][-1][0]
    current_energy = all_output[1][-1][1]
    learned_energy = all_output[1][-1][2]

    output = output.detach().cpu().numpy()
    if args.num_output_fn == 2:
        outputt = outputt.detach().cpu().numpy()
    
    gt_u = gt_u.detach().cpu().numpy()

    # Plot heatmap of kdv evolution
    plt.figure(figsize=(8, 5))
    plt.imshow(output.T, aspect='auto', extent=[args.xmin, args.xmax, args.tmin, args.tmax], cmap='viridis')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Wave Evolution via FFT')
    if save_dir:
        plt.savefig(f"{save_dir}/1d_kdv_preds_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    # Plot heatmap of kdv evolution
    plt.figure(figsize=(8, 5))
    plt.imshow(gt_u.T, aspect='auto', extent=[args.xmin, args.xmax, args.tmin, args.tmax], cmap='viridis')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Wave Evolution via FFT')
    if save_dir:
        plt.savefig(f"{save_dir}/1d_kdv_truth_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    #plot differences
    plt.figure(figsize=(8, 5))
    plt.imshow(abs(output.T - gt_u.T), aspect='auto', extent=[args.xmin, args.xmax, args.tmin, args.tmax], cmap='viridis')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Wave Evolution via FFT')
    if save_dir:
        plt.savefig(f"{save_dir}/1d_kdv_diff_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    # Plot slice in time of kdv equation ground truth and prediction
    x = np.linspace(-args.xmin, args.xmin, args.Nx)
    plt.figure(figsize=(8, 5))
    
    # Use a colorblind-friendly palette
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    num_slices = 5
    
    for k, t in enumerate(range(0, num_t, num_t//num_slices)):
        if k >= num_slices:
            break
        color = next(color_cycle)
        plt.plot(x, gt_u[t].squeeze().cpu().numpy(), 
                label=f't={t*args.t_res:.1f} GT', 
                color=color)
        plt.plot(x, output[t], 
                label=f't={t*args.t_res:.1f} Pred', 
                color=color, 
                linestyle='dashed')
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('KdV Equation: Ground Truth vs Prediction')
    plt.legend()
    if save_dir:
        plt.savefig(f"{save_dir}/1d_kdv_slices_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    

    # Plot energy over time
    plt.figure(figsize=(8, 5))
    plt.plot(example_t.detach().cpu().numpy(), true_energy.detach().cpu().numpy(), label='Ground Truth')
    if learned_energy is not None:
        plt.plot(example_t.detach().cpu().numpy(), learned_energy.detach().cpu().numpy(), label='Learned Energy', linestyle='dotted')
    if current_energy is not None:
        plt.plot(example_t.detach().cpu().numpy(), current_energy.detach().cpu().numpy(), label='Current Energy', linestyle='dotted')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Evolution')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limits to start at 0 and end slightly above max true energy
    max_true = true_energy.detach().cpu().numpy().max()
    plt.ylim(bottom=0, top=max_true*1.1)  # End 10% above max true energy
    
    if save_dir:
        plt.savefig(f"{save_dir}/1d_kdv_energy_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


# Plot wave evolution
def plot_1d_wave_evolution(args, i, data, model, save_dir=None, val=False) :

    if val:
        suffix = '_val'
    else:
        suffix = ''

    num_t = args.Nt
    x = np.linspace(-args.xmin, args.xmin, args.Nx)

    if args.method == 'deeponet':
        gt_u = data.labels[:,0,:][i*num_t:(i+1)*num_t].squeeze(-1)
        gt_ut = data.labels[:,1,:][i*num_t:(i+1)*num_t].squeeze(-1)
        example_t = data.trunk_data[i*num_t:(i+1)*num_t].to(args.device)
        example_t.requires_grad_(True)
        example_u = data.branch_data[i*num_t].view(1,-1).to(args.device)

        x=(data.branch_data[i*num_t].unsqueeze(0).repeat(num_t,1,1),)
        y=data.labels[i*num_t:(i+1)*num_t,:,:]

        all_output = model((example_u.repeat(num_t, 1), example_t.requires_grad_(True)),x=x,y=y)
        output = all_output[0]
        true_energy = all_output[1][-1][0]
        current_energy = all_output[1][-1][1]
        learned_energy = all_output[1][-1][2]

        if args.num_output_fn == 2:
            output_list = [output[:, i*int(output.shape[1]//args.num_output_fn):(i+1)*int(output.shape[1]//args.num_output_fn)] for i in range(args.num_output_fn)]
            output = output_list[0]
            outputt = output_list[1]

        output = output.detach().cpu().numpy()
        if args.num_output_fn == 2:
            outputt = outputt.detach().cpu().numpy()

        # Plot slice in time of wave equation ground truth and prediction
    
        plt.figure(figsize=(8, 5))
        
        # Use a colorblind-friendly palette
        color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        num_slices = 5
        
        for k, t in enumerate(range(0, num_t, num_t//num_slices)):
            if k >= num_slices:
                break
            color = next(color_cycle)
            plt.plot(x, gt_u[t].squeeze().cpu().numpy(), 
                    label=f't={t*args.t_res:.1f} GT', 
                    color=color)
            plt.plot(x, output[t], 
                    label=f't={t*args.t_res:.1f} Pred', 
                    color=color, 
                    linestyle='dashed')
            
        gt_u = gt_u.detach().cpu().numpy()
        gt_ut = gt_ut.detach().cpu().numpy()

    
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('Wave Equation: Ground Truth vs Prediction')
        plt.legend()
        if save_dir:
            plt.savefig(f"{save_dir}/1d_kdv_slices_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()


    elif args.method == 'full_fourier':
        gt_u = data.y[i,0]
        gt_ut = data.y[i,1]

        gt_u = gt_u.detach().cpu().numpy()
        gt_ut = gt_ut.detach().cpu().numpy()

        og_x = data.x[i,:,:]
        # if args.num_input_fn == 1:
        #         x = x[:,0,:].squeeze(-1)
        # else:
        x = og_x.view(1, -1)

        print(data.x.shape)
        print(data.y.shape)

        plt.plot(data.x[0,0,:].detach().numpy())
        plt.plot(data.y[0,0,:,0].detach().numpy())
        plt.show()



        all_output = model(x, og_x.unsqueeze(0), data.y[i,:,:,:].unsqueeze(0))

        if args.num_output_fn == 2:
            output_list = [output[:, i*int(output.shape[1]//args.num_output_fn):(i+1)*int(output.shape[1]//args.num_output_fn)] for i in range(args.num_output_fn)]
            output = output_list[0]
            outputt = output_list[1]

        example_t = torch.tensor(np.linspace(args.tmin, args.tmax, args.Nt)).to(args.device)

        output = all_output[0]
        output = output.squeeze(0).detach().cpu().numpy()
        true_energy = all_output[1][0].expand(example_t.shape)
        current_energy = all_output[1][1]
        learned_energy = all_output[1][2]
        energy_components = all_output[1][3]

        print(f'true energy shape {true_energy.shape}')
        print(f'example_t shape {example_t.shape}')

    


    # Plot heatmap of wave evolution
    plt.figure(figsize=(8, 5))
    plt.imshow(output, aspect='auto', extent=[args.xmin, args.xmax, args.tmin, args.tmax], cmap='viridis')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Wave Evolution via FFT')
    if save_dir:
        plt.savefig(f"{save_dir}/1d_wave_preds_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    # Plot heatmap of wave evolution
    plt.figure(figsize=(8, 5))
    plt.imshow(gt_u, aspect='auto', extent=[args.xmin, args.xmax, args.tmin, args.tmax], cmap='viridis')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Wave Evolution via FFT')
    if save_dir:
        plt.savefig(f"{save_dir}/1d_wave_truth_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    #plot difference
    plt.figure(figsize=(8, 5))
    plt.imshow(abs(output-gt_u), aspect='auto', extent=[args.xmin, args.xmax, args.tmin, args.tmax], cmap='viridis')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Wave Evolution via FFT')
    if save_dir:
        plt.savefig(f"{save_dir}/1d_wave_diff_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    if args.num_output_fn == 2:
        # Plot heatmap of time derivative
        plt.figure(figsize=(8, 5))
        plt.imshow(outputt, aspect='auto', extent=[args.xmin, args.xmax, args.tmin, args.tmax], cmap='viridis')
        plt.colorbar(label=r'$\partial_t u(t,x)$')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Time Derivative of the Wave Equation')
        if save_dir:
            plt.savefig(f"{save_dir}/1d_wave_dt_preds_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()


        # Plot heatmap of ground thruth time derivative
        plt.figure(figsize=(8, 5))
        plt.imshow(gt_ut, aspect='auto', extent=[args.xmin, args.xmax, args.tmin, args.tmax], cmap='viridis')
        plt.colorbar(label=r'$\partial_t u(t,x)$')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Time Derivative of the Wave Equation')
        if save_dir:
            plt.savefig(f"{save_dir}/1d_wave_dt_ground_truth_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        #plot differences
        plt.figure(figsize=(8, 5))
        plt.imshow(abs(outputt-gt_ut), aspect='auto', extent=[args.xmin, args.xmax, args.tmin, args.tmax], cmap='viridis')
        plt.colorbar(label=r'$\partial_t u(t,x)$')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Time Derivative of the Wave Equation')
        if save_dir:
            plt.savefig(f"{save_dir}/1d_wave_dt_diff_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

    # Plot energy over time
    plt.figure(figsize=(8, 5))
    # plt.plot(example_t.detach().cpu().numpy(), energy.detach().cpu().numpy(), label='Energy')
    # plt.plot(example_t.detach().cpu().numpy(), energy_learned_new.detach().cpu().numpy(), label='Energy')
    plt.plot(example_t.detach().cpu().numpy(), true_energy.T.detach().cpu().numpy(), label='Ground Truth')
    plt.plot(example_t.detach().cpu().numpy(), energy_components['target_energy'].T.detach().cpu().numpy(), label='Target Energy')
    plt.plot(example_t.detach().cpu().numpy(), energy_components['og_target_energy'].T.detach().cpu().numpy(), label='OG Target Energy')
    if learned_energy is not None:
        plt.plot(example_t.detach().cpu().numpy(), learned_energy.T.detach().cpu().numpy(), label='Learned Energy', linestyle='dotted')
    if current_energy is not None:
        plt.plot(example_t.detach().cpu().numpy(), current_energy.T.detach().cpu().numpy(), label='Current Energy', linestyle='dotted')
    # plt.plot(example_t.detach().cpu().numpy(), current_energy.detach().cpu().numpy(), label='Returned Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Evolution')
    plt.grid(True)
    plt.legend()
    if save_dir:
        plt.savefig(f"{save_dir}/1d_wave_energy_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    #plot the ut components
    plt.figure(figsize=(8, 5))
    # plt.plot(example_t.detach().cpu().numpy(), energy.detach().cpu().numpy(), label='Energy')
    # plt.plot(example_t.detach().cpu().numpy(), energy_learned_new.detach().cpu().numpy(), label='Energy')
    plt.plot(example_t.detach().cpu().numpy(), energy_components['true_energy_u_component'].expand(example_t.shape).T.detach().cpu().numpy(), label='Ground Truth ux Component')
    plt.plot(example_t.detach().cpu().numpy(), energy_components['target_energy_ux_component'].T.detach().cpu().numpy(), label='Target ux Component')
    plt.plot(example_t.detach().cpu().numpy(), energy_components['og_target_energy_ux_component'].T.detach().cpu().numpy(), label='OG Target ux Component')

    if learned_energy is not None:
        plt.plot(example_t.detach().cpu().numpy(), energy_components['learned_energy_u_component'].T.detach().cpu().numpy(), label='Learned Energy ux Component', linestyle='dotted')
    if current_energy is not None:
        plt.plot(example_t.detach().cpu().numpy(), energy_components['current_energy_u_component'].T.detach().cpu().numpy(), label='Current Energy ux Component', linestyle='dotted')
    # plt.plot(example_t.detach().cpu().numpy(), current_energy.detach().cpu().numpy(), label='Returned Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Evolution of ux Component')
    plt.grid(True)
    plt.legend()
    if save_dir:
        plt.savefig(f"{save_dir}/1d_wave_energy_ux_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


    #plot the ut components
    plt.figure(figsize=(8, 5))
    # plt.plot(example_t.detach().cpu().numpy(), energy.detach().cpu().numpy(), label='Energy')
    # plt.plot(example_t.detach().cpu().numpy(), energy_learned_new.detach().cpu().numpy(), label='Energy')
    plt.plot(example_t.detach().cpu().numpy(), energy_components['true_energy_ut_component'].expand(example_t.shape).T.detach().cpu().numpy(), label='Ground Truth ut Component')
    plt.plot(example_t.detach().cpu().numpy(), energy_components['target_energy_ut_component'].T.detach().cpu().numpy(), label='Target ut Component')
    plt.plot(example_t.detach().cpu().numpy(), energy_components['og_target_energy_ut_component'].T.detach().cpu().numpy(), label='OG Target ut Component')
    if learned_energy is not None:
        plt.plot(example_t.detach().cpu().numpy(), energy_components['learned_energy_ut_component'].T.detach().cpu().numpy(), label='Learned Energy ut Component', linestyle='dotted')
    if current_energy is not None:
        plt.plot(example_t.detach().cpu().numpy(), energy_components['current_energy_ut_component'].T.detach().cpu().numpy(), label='Current Energy ut Component', linestyle='dotted')
    # plt.plot(example_t.detach().cpu().numpy(), current_energy.detach().cpu().numpy(), label='Returned Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Evolution of ut Component')
    plt.grid(True)
    plt.legend()
    if save_dir:
        plt.savefig(f"{save_dir}/1d_wave_energy_ut_{i}_{suffix}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    







    