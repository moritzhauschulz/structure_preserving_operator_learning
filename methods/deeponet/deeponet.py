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

    epochs = args.epochs
    
    train_data = data[0]
    val_data = data[1]
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

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
        if isinstance(preds, tuple):
            (preds, aux) = preds
        else:
            aux = None


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

        # print(f'args.track_all_losses is {args.track_all_losses}')

        # if args.track_all_losses:     #TODO: fix this
        #     aux_loss, aux_losses = compute_aux_loss(args,aux)
        #     losses.update(aux_losses)
        #     nrg_loss = compute_nrg_loss(args, x, y, preds)
        #     losses['nrg_loss'] = nrg_loss

        #     if args.loss == 'reg':
        #         loss += aux_loss
        #     elif args.loss == 'nrg':
        #         loss += nrg_loss
        #     elif not args.loss == 'mse':
        #         raise ValueError(f'Loss {args.loss} not recognized')
        # else:
        #     if args.loss == 'reg':
        #         aux_loss, aux_losses = compute_aux_loss(args, aux)
        #         loss += aux_loss
        #         losses.update(aux_losses)
        #     elif args.loss == 'nrg':
        #         nrg_loss = compute_nrg_loss(args,x, y, preds)
        #         loss += nrg_loss
        #         losses['nrg_loss'] = nrg_loss
        #     elif not args.loss == 'mse':
        #         raise ValueError(f'Loss {args.loss} not recognized')
            
        return main_loss, losses

    def compute_nrg_loss(args,x, y, preds):
        # print(y.shape)
        # print('computing energy loss')
        nrg = (0.5 * (x[0][:,2] * x[0][:,0]) ** 2 + 0.5 * x[0][:,1] ** 2)
        nrg_hat = (0.5 * (x[0][:,2] * preds[:,0]) ** 2 + 0.5 * preds[:,1] ** 2)
        nrg_loss = mse_loss(nrg, nrg_hat)
        # if args.wandb :
        #         wandb.log({'iteration': i, 'nrg_loss': nrg_loss})
        return args.nrg_weight * nrg_loss

    def compute_aux_loss(args,aux):
        (branch_out, x_loc) = aux
        #compute b^t * b to check orthonormality
        branch_out = branch_out.view(-1, args.num_outputs, branch_out.shape[1] // args.num_outputs)
        branch_orthonormality = torch.bmm(branch_out.permute(0, 2, 1) , branch_out)
        trunk_normality = torch.norm(x_loc, p=2, dim=1, keepdim=True)
        branch_orthonormality_loss = torch.mean((branch_orthonormality - torch.eye(branch_out.shape[2])) ** 2)
        trunk_normality_loss = torch.mean((trunk_normality -1) ** 2)
        # if args.wandb :
        #     wandb.log({'iteration': i, 'branch_orthonormality_loss': branch_orthonormality_loss, 'trunk_normality_loss': trunk_normality_loss})
        return args.branch_weight * branch_orthonormality_loss + args.trunk_weight * trunk_normality_loss, {'branch_orthonormality_loss': branch_orthonormality_loss, 'trunk_normality_loss': trunk_normality_loss}

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None
    best_model_epoch = None
    
    pbar = tqdm(range(1,epochs + 1))
    for i in pbar:
        train_loss = 0

        model.epoch = i

        epoch_losses = {}
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
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())


            val_loss = 0


            for batch_idx, (x, y) in enumerate(val_loader):
                x[0] = x[0].to(args.device)
                x[1] = x[1].to(args.device)
                
                loss, losses = compute_loss(args,i, model, x, y)
                val_loss += loss.item()
                for key, value in losses.items():
                    key = f'{key}_val'
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())

            pbar.set_description(f'epoch {i}; loss {train_loss}; val_loss {val_loss}; train_val_loss {train_val_loss}')
            val_losses.append(val_loss)
            train_losses.append(train_loss)
            with open(f'{args.save_models}/losses.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
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

    if args.save_model:
        torch.save(model.state_dict(), args.save_models + '/final_ckpt.pth')
        if best_model_state is not None:
            torch.save(best_model_state, args.save_models + f'/best_ckpt_epoch_{best_model_epoch}.pth')

    print(f'Training completed. Best model at epoch {best_model_epoch}')
    print('Generating examples...')

    if args.wandb:
        wandb.finish()
    
    # visualize_loss(args, train_losses, val_losses)
    wandb_viz_loss(args.exp_n, args.save_plots)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if args.problem == 'harmonic_oscillator':
        for i in range(args.num_examples): 
            examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad = compute_example_with_energy(i, args, train_data, model)
            visualize_example_with_energy('train', args, examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad)
        for i in range(args.num_examples): 
            examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad = compute_example_with_energy(i, args, val_data, model)
            visualize_example_with_energy('val', args, examples, example_t, ground_truth, output, nrg_hat, vel_nrg_hat, numerical_nrg, nrg, grad)
    elif args.problem == '1d_KdV_Soliton':
        # h = 0.5
        # a = (args.IC['a'][0] +  args.IC['a'][1])/2
        # c = (args.IC['c'][0] +  args.IC['c'][1])/2

        # if args.use_ifft:
        #     plot_1d_KdV_Soliton_ifft(args, h,0.001, a, c, model, save_dir=args.save_plots)
        # else:
        #     plot_1d_KdV_Soliton(args, h,0.001, a, c, model, save_dir=args.save_plots)
        for i in range(args.num_examples):
            plot_1d_KdV_evolution(args, i, train_data, model, save_dir=args.save_plots)
        for i in range(args.num_examples):
            plot_1d_KdV_evolution(args, i, val_data, model, save_dir=args.save_plots, val=True)
    elif args.problem == '1d_wave':
        model.eval()
        for i in range(args.num_examples):
            plot_1d_wave_evolution(args, i, train_data, model, save_dir=args.save_plots)
        for i in range(args.num_examples):
            plot_1d_wave_evolution(args, i, val_data, model, save_dir=args.save_plots, val=True)




        


