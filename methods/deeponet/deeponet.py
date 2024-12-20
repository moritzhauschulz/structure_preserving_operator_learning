import numpy as np
from scipy import io
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os
from itertools import cycle
from deepxde.nn.pytorch.deeponet import DeepONet
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from deepxde.optimizers.pytorch.optimizers import _get_learningrate_scheduler

from utils.utils import visualize_example, visualize_example_with_energy, plot_y_with_initial_conditions, visualize_loss, print_summary
from utils.model import DeepONetWithGrad
from utils.data import harmonic_oscillator
from utils.data import get_data
from utils.utils import compute_example_with_energy
import csv

np.random.seed(42)

def main_loop(args, data):
    epochs = args.epochs
    
    train_data = data[0]
    val_data = data[1]
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = DeepONetWithGrad(args.branch_layers, args.trunk_layers, args.deepo_activation, "Glorot normal", num_outputs=2)
    if args.load_checkpoint is not None:
        model.load_state_dict(torch.load(args.load_model))

    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=0.0)
    decay = ("inverse time", epochs // 5, 0.5)
    scheduler = _get_learningrate_scheduler(optimizer, decay)

    compute_loss = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    
    pbar = tqdm(range(epochs))
    for i in pbar:
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            
            preds = model(x)

            loss = compute_loss(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if i % args.eval_every == 0:
            val_loss = 0
            for batch_idx, (x, y) in enumerate(val_loader):
                preds = model(x)
                val_loss += compute_loss(preds, y).item()
            train_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                preds = model(x)
                train_loss += compute_loss(preds, y).item()
            pbar.set_description(f'epoch {i}; loss {train_loss}; val_loss {val_loss}')
            val_losses.append(val_loss)
            train_losses.append(train_loss)
            with open(os.path.join(args.save_dir, 'losses.csv'), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
                writer.writerow([i, train_loss, val_loss])

        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), args.save_dir + '/ckpt.pth')

    print('Training completed.')
    print('Generating examples...')
    
    visualize_loss(args, train_losses, val_losses)

    examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, nrg = compute_example_with_energy(args, val_data, model)
    visualize_example_with_energy(args, examples, example_t, ground_truth, output, nrg, nrg_hat, vel_nrg_hat)
