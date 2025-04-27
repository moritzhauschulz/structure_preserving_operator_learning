SCRIPT_NAME=$(basename "$0" .sh)

# Run deeponet on harmonic oscillator problem
# python main.py --problem harmonic_oscillator --device cpu --method deeponet --branch_layers 3 128 128 128 4 --trunk_layers 1 128 128 2 --epochs 250 --load_data False --multi_output_strategy orthonormal_split_branch --num_outputs 2 --experiment_name "$SCRIPT_NAME"

#wider net
python main.py --problem harmonic_oscillator --device cpu --method deeponet --branch_layers 3 256 245 256 4 --trunk_layers 1 256 256 256 2 --epochs 250 --load_data False --loss reg --multi_output_strategy orthonormal_branch_normal_trunk_reg --num_outputs 2 --experiment_name "$SCRIPT_NAME"
