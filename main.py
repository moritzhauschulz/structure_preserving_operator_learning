import os
import argparse
import json 
import datetime
from utils.data import get_data

from methods.deeponet.deeponet import main_loop as deeponet_main_loop

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')
    parser.add_argument('--method', type=str, default='deeponet', 
        choices=[
            'deeponet'
        ],
    )
    parser.add_argument('--tmax', type=int, default=25, help='Maximum time')
    parser.add_argument('--tmin', type=int, default=0, help='Minimum time')
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
    temp_args, _ = parser.parse_known_args()
    
    #optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--decay', type=tuple, default=('inverse time', temp_args.epochs // 5, 0.5), help='Decay')

    #deeponet
    parser.add_argument('--n_branch', type=int, default=50, help='Number of branches')
    parser.add_argument('--n_trunk', type=int, default=1000, help='Number of trunks')
    parser.add_argument('--branch_layers', type=list, default=[3, 128, 128, 128, 128], help='Branch layers')
    parser.add_argument('--trunk_layers', type=list, default=[1, 128, 128, 128], help='Trunk layers')
    parser.add_argument('--deepo_activation', type=str, default='tanh', help='Trunk layers')

    args = parser.parse_args()
    
    #data
    if args.problem == 'harmonic_oscillator':
        if temp_args.IC is None:
            args.IC = {'q0': [-1,1], 'p0': [-1,1], 'omega': [0,1]}

    #log
    args.exp_n = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.save_dir = f'./methods/{args.method}/experiments/exp_n_{args.exp_n}'
    args.data_dir = f'./data/{args.problem}'
    args.data_config = f'_q0_{args.IC["q0"]}_p0_{args.IC["p0"]}_omega_{args.IC["omega"]}_tmin_{args.tmin}_tmax_{args.tmax}.pkl'
    return args

def log_args(args):
    os.makedirs(args.save_dir, exist_ok=True)
    with open(f'{args.save_dir}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    

if __name__ == '__main__':
    args = get_args()
    log_args(args)
    data = get_data(args)
    print(f'Saving experiment data to {args.save_dir}')
    main_fn = eval(f'{args.method}_main_loop')
    main_fn(args, data)

