import os
import argparse
import json 
import datetime
import torch
from utils.data import get_data
import wandb
import json

from methods.deeponet.deeponet import main_loop as deeponet_main_loop
from methods.full_fourier.full_fourier import main_loop as full_fourier_main_loop
from filelock import FileLock

def get_args():
    print('getting args')
    parser = argparse.ArgumentParser(description='Pipeline')
    parser.add_argument('--method', type=str, default='deeponet', 
        choices=[
            'deeponet',
            'full_fourier',
        ],
    )
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device')
    parser.add_argument('--tmax', type=float, default=25, help='Maximum time')
    parser.add_argument('--tmin', type=float, default=0, help='Minimum time')
    parser.add_argument('--t_res', type=float, default=0.1, help='Time resolution')
    parser.add_argument('--xmax', type=int, default=25, help='Maximum x')
    parser.add_argument('--xmin', type=int, default=-25, help='Minimum x')
    parser.add_argument('--x_res', type=float, default=0.1, help='x resolution')
    parser.add_argument('--Nx', type=int, default=49, help='Number of Collocation Points in Space')
    parser.add_argument('--Nt', type=int, default=49, help='Number of Collocation Points in Time')
    parser.add_argument('--load_data', type=bool, default=True, help='Load data')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Load model')
    parser.add_argument('--save_data', type=bool, default=True, help='Save data')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model')
    parser.add_argument('--IC', type=str, default=None, help='Initial Conditions')
    parser.add_argument('--problem', type=str, default='harmonic_oscillator', help='Problem name')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1028, help='Batch size')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every')
    parser.add_argument('--num_outputs', type=int, default=1, help='Number of outputs')
    parser.add_argument('--num_examples', type=int, default=3, help='Number of examples')
    parser.add_argument('--branch_weight', type=float, default=1, help='weight on orthonormality loss')
    parser.add_argument('--trunk_weight', type=float, default=1, help='weight on normality loss')
    parser.add_argument('--nrg_weight', type=float, default=None, help='weight on nrg loss')
    parser.add_argument('--loss_weights', type=list, default=None, help='weight on nrg loss')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 128, 128], help='Branch layers')
    parser.add_argument('--inference_norm', type=bool, default=False, help='True if only norming at infrence time')

    temp_args, _ = parser.parse_known_args()

    #data gen
    parser.add_argument('--data_dt', type=float, default=0.01, help='weight on nrg loss')
    parser.add_argument('--data_modes', type=int, default=10, help='weight on nrg loss')
    parser.add_argument('--zero_zero_mode', type=bool, default=False, help='force zero mode to zero')

    #optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--decay', type=tuple, default=('inverse time', temp_args.epochs // 5, 0.5), help='Decay')
   
    #fourier related
    parser.add_argument('--num_input_fn', type=int, default=1, help='Number of input functions')
    parser.add_argument('--num_output_fn', type=int, default=1, help='Number of output functions')
    parser.add_argument('--fourier_input', type=bool, default=False, help='Fourier input')
    parser.add_argument('--use_ifft', type=bool, default=False, help='Whether to use ifft – only relevant for some problems')
    parser.add_argument('--activation', type=str, default='swish', help='Trunk layers')
    parser.add_argument('--num_inputs', type=int, default=1, help='Number of inputs')
    parser.add_argument('--t_filter_cutoff_ratio', type=float, default=1, help='Filter cutoff ratio for time derivative')
    parser.add_argument('--x_filter_cutoff_ratio', type=float, default=1, help='Filter cutoff ratio for time derivative')


    #deeponet
    parser.add_argument('--n_branch', type=int, default=50, help='Number of branches')
    parser.add_argument('--n_trunk', type=int, default=1000, help='Number of trunks')
    parser.add_argument('--branch_layers', type=int, nargs='+', default=[3, 128, 128, 128, 4], help='Branch layers')
    parser.add_argument('--trunk_layers', type=int, nargs='+', default=[1, 128, 128, 2], help='Trunk layers')
    parser.add_argument('--deepo_activation', type=str, default='tanh', help='Trunk layers')
    parser.add_argument('--strategy', type=str, default=None, choices={'independent','split_both','split_branch','split_trunk','orthonormal_branch_normal_trunk', 'normal_trunk', 'orthonormal_trunk', 'vanilla', 'QR', 'Fourier', 'FourierQR', 'FourierNorm', 'FullFourier', 'FullFourierNorm', 'FullFourierAvgNorm', 'normal'}, help='DeepONet strategy')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'reg', 'nrg'], help='Loss function')

    #nrg
    parser.add_argument('--use_implicit_nrg', type=bool, default=False, help='Use implicit nrg')
    parser.add_argument('--num_norm_refinements', type=int, default=1, help='Number of refinements')
    parser.add_argument('--detach', type=bool, default=False, help='Detach?')


    #wanbd
    parser.add_argument('--wandb_user', type=str, default='moritz-hasuschulz', help='Wandb user')
    parser.add_argument('--wandb', type=bool, default=True, help='Wandb')
    parser.add_argument('--wandb_project', type=str, default='structure-preserving-operator-learning', help='Wanbd project name')
    parser.add_argument('--track_all_losses', type=bool, default=False, help='Track all losses')


    args = parser.parse_args()

    if args.IC is not None:
        args.IC = json.loads(args.IC)

    #adjust architecture
    if args.fourier_input:
        args.branch_layers[0] = args.Nx * args.num_input_fn
    
    args.branch_layers *= args.num_input_fn

    if 'Fourier' in args.strategy and args.method == 'deeponet':
        args.num_outputs = int((args.Nx + 1)/2) * args.num_output_fn
        print(f'Automatically adjusted num_outputs for Fourier to {args.num_outputs}')
        args.branch_layers[-1] = args.num_outputs * args.trunk_layers[-1] * 2
        print(f'Automatically adjusted branch ({args.branch_layers}) and trunk layers ({args.trunk_layers}) for Fourier')
        print('Automatically adjusted branch and trunk layers for Fourier')

    if args.method == 'full_fourier':
        args.num_inputs = args.num_input_fn * args.Nx
        print(f'Automatically adjusted num_inputs for full fourier to {args.num_inputs}')
        args.num_outputs = args.num_output_fn * args.Nx * (args.Nt //2 + 1) * 2 # *2 for complex

    
    
    #data
    if args.problem == 'harmonic_oscillator':
        if temp_args.IC is None:
            args.IC = {'q0': [-1,1], 'p0': [-1,1], 'omega': [1,1]}
        args.data_config = f'_q0_{args.IC["q0"]}_p0_{args.IC["p0"]}_omega_{args.IC["omega"]}_tmin_{args.tmin}_tmax_{args.tmax}_num_out_{args.num_outputs}_nbranch_{args.n_branch}.pkl'

    if args.problem == '1d_KdV_Soliton':
        if temp_args.IC is None:
            args.IC = {'c': [2.5,2.5], 'a': [-0,0]} #make this harder
        args.data_config = f'_c_{args.IC["c"]}_a_{args.IC["a"]}_tmin_{args.tmin}_tmax_{args.tmax}_tres_{args.t_res}_xmin_{args.xmin}_xmax_{args.xmax}_xres_{args.x_res}_num_out_{args.num_outputs}_nbranch_{args.n_branch}.pkl'

    if args.problem == '1d_wave':
        if temp_args.IC is None:
            args.IC = {'c': 5, 'type': 'periodic_gp', 'params': {'lengthscale':0.5, 'variance':1.0}} #make this harder
        args.data_config = f'_c_{args.IC["c"]}_tmin_{args.tmin}_tmax_{args.tmax}_tres_{args.t_res}_xmin_{args.xmin}_xmax_{args.xmax}_xres_{args.x_res}_num_out_{args.num_outputs}_nbranch_{args.n_branch}_data_dt_{args.data_dt}.pkl'

    if args.problem == '1d_wave' and args.method == 'full_fourier':
        time_period = (args.xmax - args.xmin)/args.IC['c']
        print('Forced time domain length to equal time periodicity')
        args.tmax = args.tmin + time_period

    if args.loss_weights is None:
        args.loss_weights = [1 for i in range(args.num_output_fn)]

    if args.method == 'deeponet':
        args.multi_output_strategy = args.strategy

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
    # assert args.load_data == False, 'Should make new data always'
    # assert args.use_ifft == True, 'Should use IFFT'


    args_dict = log_args(args)
    data = get_data(args)
    print(args.num_outputs)

    print(f'Saving experiment data to {args.save_dir}')
    main_fn = eval(f'{args.method}_main_loop')
    main_fn(args, data)

