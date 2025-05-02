import numpy as np
from scipy import io
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os
from itertools import cycle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from deepxde.optimizers.pytorch.optimizers import _get_learningrate_scheduler

from utils.viz import visualize_example, visualize_example_with_energy, plot_y_with_initial_conditions, visualize_loss, compute_example_with_energy, wandb_viz_loss, plot_1d_KdV_Soliton, plot_1d_KdV_Soliton_ifft, plot_1d_wave_evolution, plot_1d_KdV_evolution
from utils.model import DeepONetWithGrad
from utils.data import harmonic_oscillator
from utils.data import get_data

import csv
import wandb

np.random.seed(42)


def mse_complex(x, y):
    return torch.mean(torch.abs(x - y) ** 2)  # Squared magnitude error


def main_loop(args, data):

    if args.wandb:
        args_dict = vars(args)
        wandb.init(project=args.wandb_project, config=args_dict, id=args.exp_n)
    else:
        print('wandb is disabled, some functionality will be disabled')

    epochs = args.epochs

    train_data = data[0]
    val_data = data[1]
    if args.test_set:
        test_data = data[2]

    
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    if args.test_set:
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = DeepONetWithGrad(args, args.branch_layers, args.trunk_layers, args.deepo_activation, "Glorot normal", num_outputs=args.num_outputs, multi_output_strategy=args.multi_output_strategy)
    if args.load_checkpoint is not None:
        model.load_state_dict(torch.load(args.load_checkpoint))
    model.to(args.device)

    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=0.0)
    decay = ("inverse time", epochs // 5, 0.5)
    scheduler = _get_learningrate_scheduler(optimizer, decay)

    mse_loss = torch.nn.MSELoss()

    def compute_loss(args,i,model,x,y,log=True):

        og_y = y
        og_x = x.copy()


        if args.problem == '1d_wave':
            if args.num_input_fn == 1:
                x[0] = x[0][:,0,:].squeeze(-1)
            else:
                x[0] = x[0].view(x[0].shape[0], -1)
            if args.num_output_fn == 1:
                y = y[:,0,:].squeeze(-1)
            else:
                y = y.view(y.shape[0], -1)


        y = y.to(args.device)

        preds = model(x, og_x, og_y)
        # if isinstance(preds, tuple):
        preds, aux = preds
        # else:
        #     aux = None


        losses = {}
        main_loss = torch.tensor(0.0, device=args.device)
        block_n = preds.shape[1]//args.num_output_fn
        if torch.is_complex(preds):
            for i in range(args.num_output_fn):
                losses[f'loss_{i}'] = mse_complex(preds[:,block_n*i:block_n*(i+1)], y[:,block_n*i:block_n*(i+1)])
                main_loss += losses[f'loss_{i}']
                print(losses[f'loss_{i}'])
        else:
            for i in range(args.num_output_fn):
                losses[f'loss_{i}'] = mse_loss(preds[:,block_n*i:block_n*(i+1)], y[:,block_n*i:block_n*(i+1)])
                main_loss += losses[f'loss_{i}']
        losses['mse_loss'] = main_loss

        if aux is not None: #aux is not None:
            energies = aux
            true_energy, current_energy, learned_energy, energy_components = energies
            losses['current_energy_loss'] = mse_loss(true_energy, current_energy)
            losses['learned_energy_loss'] = mse_loss(true_energy, learned_energy)
            if energy_components is not None:
                losses['current_ux_energy_loss'] = mse_loss(energy_components['target_energy_ux_component'], energy_components['current_energy_u_component'].squeeze(0))
                losses['current_ut_energy_loss'] = mse_loss(energy_components['target_energy_ut_component'], energy_components['current_energy_ut_component'].squeeze(0))
                losses['learned_ux_energy_loss'] = mse_loss(energy_components['target_energy_ux_component'], energy_components['learned_energy_u_component'].squeeze(0))
                losses['learnedt_ut_energy_loss'] = mse_loss(energy_components['target_energy_ut_component'], energy_components['learned_energy_ut_component'].squeeze(0))


    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None
    best_model_epoch = None

    if args.eval_only:
        epochs = 1
    
    pbar = tqdm(range(1,epochs + 1))


    for i in pbar:
        train_loss = 0

        model.epoch = i

        epoch_losses = {}

        if not args.eval_only:
            for batch_idx, (x, y) in enumerate(train_loader):
                x[0] = x[0].to(args.device)
                x[1] = x[1].to(args.device)



                loss, losses = compute_loss(args,i, model, x, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()


        if i % args.eval_every == 0 or i ==1:
            train_val_loss = 0

            model.eval()

            for batch_idx, (x, y) in enumerate(train_loader):
                x[0] = x[0].to(args.device)
                x[1] = x[1].to(args.device)
                
                loss, losses = compute_loss(args,i, model, x, y)
                train_val_loss += loss.item()
                for key, value in losses.items():
                    key = f'{key}_train'
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value.item()


            val_loss = 0


            for batch_idx, (x, y) in enumerate(val_loader):
                x[0] = x[0].to(args.device)
                x[1] = x[1].to(args.device)
                
                loss, losses = compute_loss(args,i, model, x, y)
                val_loss += loss.item()
                for key, value in losses.items():
                    key = f'{key}_val'
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value.item()

            pbar.set_description(f'epoch {i}; loss {train_loss}; val_loss {val_loss}; train_val_loss {train_val_loss}')
            val_losses.append(val_loss)
            train_losses.append(train_loss)
            with open(f'{args.save_models}/losses.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if i == 1:
                    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
                writer.writerow([i, train_val_loss, val_loss])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                best_model_epoch = i

            model.train()

        for key, value in epoch_losses.items():
                epoch_losses[key] = np.mean(value)
        
        if args.wandb:
            wandb.log({'epoch': i, **epoch_losses})

        scheduler.step()

    if args.save_model and not args.eval_only:
        torch.save(model.state_dict(), args.save_models + '/final_ckpt.pth')
        if best_model_state is not None:
            torch.save(best_model_state, args.save_models + f'/best_ckpt_epoch_{best_model_epoch}.pth')

    print(f'Training completed. Best model at epoch {best_model_epoch}')
    print('Generating examples...')

    
    # visualize_loss(args, train_losses, val_losses)
    if args.wandb:
        wandb_viz_loss(args.exp_n, args.save_plots)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if args.test_set:
        test_losses = {}
        test_loss = 0
        for batch_idx, (x, y) in enumerate(test_loader):
            x[0] = x[0].to(args.device)
            x[1] = x[1].to(args.device)
            
            loss, losses = compute_loss(args,i, model, x, y)
            test_loss += loss.item()
            for key, value in losses.items():
                key = f'{key}_test'
                if key not in test_losses:
                    test_losses[key] = 0
                test_losses[key] += value.item()
            
        if args.wandb:
            wandb.log({'epoch': i, **test_losses})
        print(f'Test loss: {test_loss}')

    if args.wandb:
        wandb.finish()
    

    if args.problem == 'harmonic_oscillator':
        for i in range(args.num_examples): 
            examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad = compute_example_with_energy(i, args, train_data, model)
            visualize_example_with_energy('train', args, examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad)
        for i in range(args.num_examples): 
            examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad = compute_example_with_energy(i, args, val_data, model)
            visualize_example_with_energy('val', args, examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad)
        for i in range(args.num_examples): 
            examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad = compute_example_with_energy(i, args, test_data, model)
            visualize_example_with_energy('test', args, examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad)
    elif args.problem == '1d_KdV_Soliton':
        for i in range(args.num_examples):
            plot_1d_KdV_evolution(args, i, train_data, model, save_dir=args.save_plots)
        for i in range(args.num_examples):
            plot_1d_KdV_evolution(args, i, val_data, model, save_dir=args.save_plots, val=True)
        for i in range(args.num_examples):
            plot_1d_KdV_evolution(args, i, test_data, model, save_dir=args.save_plots, val=True)
    elif args.problem == '1d_wave':
        model.eval()
        for i in range(args.num_examples):
            plot_1d_wave_evolution(args, i, train_data, model, save_dir=args.save_plots)
        for i in range(args.num_examples):
            plot_1d_wave_evolution(args, i, val_data, model, save_dir=args.save_plots, suffix='_val')
        for i in range(args.num_examples):
            plot_1d_wave_evolution(args, i, test_data, model, save_dir=args.save_plots, suffix='_test')




        


