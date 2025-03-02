import os
import argparse
import json 
import datetime
import torch
from utils.data import get_data
import wandb

from methods.deeponet.deeponet import main_loop as deeponet_main_loop
from filelock import FileLock

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')
    parser.add_argument('--method', type=str, default='deeponet', 
        choices=[
            'deeponet'
        ],
    )
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device')
    parser.add_argument('--tmax', type=int, default=25, help='Maximum time')
    parser.add_argument('--tmin', type=int, default=0, help='Minimum time')
    parser.add_argument('--t_res', type=int, default=0.1, help='Time resolution')
    parser.add_argument('--xmax', type=int, default=25, help='Maximum time')
    parser.add_argument('--xmin', type=int, default=-25, help='Minimum time')
    parser.add_argument('--x_res', type=float, default=0.1, help='x resolution')
    parser.add_argument('--load_data', type=bool, default=True, help='Load data')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Load model')
    parser.add_argument('--save_data', type=bool, default=True, help='Save data')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model')
    parser.add_argument('--IC', type=dict, default=None, help='Initial Conditions')
    parser.add_argument('--problem', type=str, default='harmonic_oscillator', help='Problem name')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every')
    parser.add_argument('--num_outputs', type=int, default=1, help='Number of outputs')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of examples')
    parser.add_argument('--branch_weight', type=float, default=1, help='weight on orthonormality loss')
    parser.add_argument('--trunk_weight', type=float, default=1, help='weight on normality loss')
    parser.add_argument('--nrg_weight', type=float, default=1, help='weight on nrg loss')

    temp_args, _ = parser.parse_known_args()
    

    #optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--decay', type=tuple, default=('inverse time', temp_args.epochs // 5, 0.5), help='Decay')

    #deeponet
    parser.add_argument('--n_branch', type=int, default=50, help='Number of branches')
    parser.add_argument('--n_trunk', type=int, default=1000, help='Number of trunks')
    parser.add_argument('--branch_layers', type=int, nargs='+', default=[3, 128, 128, 128, 4], help='Branch layers')
    parser.add_argument('--trunk_layers', type=int, nargs='+', default=[1, 128, 128, 2], help='Trunk layers')
    parser.add_argument('--deepo_activation', type=str, default='tanh', help='Trunk layers')
    parser.add_argument('--multi_output_strategy', type=str, default=None, choices={'independent','split_both','split_branch','split_trunk','orthonormal_branch_normal_trunk', 'normal_trunk', 'orthonormal_trunk', 'orthonormal_branch_normal_trunk_reg', 'QR', 'Fourier'}, help='DeepONet strategy')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'reg', 'nrg'], help='Loss function')
    

    #wanbd
    parser.add_argument('--wandb_user', type=str, default='moritz-hasuschulz', help='Wandb user')
    parser.add_argument('--wandb', type=bool, default=True, help='Wandb')
    parser.add_argument('--wandb_project', type=str, default='structure-preserving-operator-learning', help='Wanbd project name')
    parser.add_argument('--track_all_losses', type=bool, default=False, help='Track all losses')


    args = parser.parse_args()

    print(args.num_outputs)
    
    #data
    if args.problem == 'harmonic_oscillator':
        if temp_args.IC is None:
            args.IC = {'q0': [-1,1], 'p0': [-1,1], 'omega': [1,1]}
        args.data_config = f'_q0_{args.IC["q0"]}_p0_{args.IC["p0"]}_omega_{args.IC["omega"]}_tmin_{args.tmin}_tmax_{args.tmax}_num_out_{args.num_outputs}.pkl'

    if args.problem == '1d_KdV_Soliton':
        if temp_args.IC is None:
            args.IC = {'c': [1,5], 'a': [-10,10]}
        args.data_config = f'_c_{args.IC["c"]}_a_{args.IC["a"]}_tmin_{args.tmin}_tmax_{args.tmax}_tres_{args.t_res}_xmin_{args.xmin}_xmax_{args.xmax}_xres_{args.x_res}_num_out_{args.num_outputs}.pkl'

    #log
    args.exp_n = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.save_dir = f'./methods/{args.method}/experiments/{args.experiment_name}/exp_n_{args.exp_n}'
    args.data_dir = f'./data/{args.problem}'
    args.save_models = f'{args.save_dir}/models'
    args.save_plots = f'{args.save_dir}/plots'
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_models, exist_ok=True)
    os.makedirs(args.save_plots, exist_ok=True)
    
    #device
    if torch.backends.mps.is_available() and args.device == 'mps':
        print("MPS backend is available!")
        device = torch.device("mps")
        args.device = device
    elif torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA is available!")
        device = torch.device("cuda")
        args.device = device
    else:
        print("mps/cuda backend is not available or not selected.")
        device = torch.device("cpu")
        args.device == device
    print(f"Using device: {device}")

    return args

def log_args(args):
    with open(f'{args.save_dir}/args.json', 'w') as f:
        args_dict = args.__dict__.copy()
        args_dict['device'] = str(args.device)
        json.dump(args_dict, f, indent=2)
    log_args_to_dict(args)
    return args_dict

def log_args_to_dict(args):

    overall_log_path = f'./methods/{args.method}/experiments/{args.experiment_name}/overall_log.json'
    lock_path = f'{overall_log_path}.lock'
    
    with FileLock(lock_path):
        if os.path.exists(overall_log_path):
            with open(overall_log_path, 'r') as f:
                overall_log = json.load(f)
        else:
            overall_log = {}
        
        args_dict = args.__dict__.copy()
        args_dict['device'] = str(args.device)
        overall_log[args.exp_n] = args_dict

        with open(overall_log_path, 'w') as f:
            json.dump(overall_log, f, indent=2)

if __name__ == '__main__':
    args = get_args()
    args_dict = log_args(args)
    data = get_data(args)
    print(args.num_outputs)

    print(f'Saving experiment data to {args.save_dir}')
    main_fn = eval(f'{args.method}_main_loop')
    main_fn(args, data)

