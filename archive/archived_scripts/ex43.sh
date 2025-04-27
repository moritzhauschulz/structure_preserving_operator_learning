SCRIPT_NAME=$(basename "$0" .sh)

# Run deeponet on harmonic oscillator problem
# python main.py --problem harmonic_oscillator --device cpu --method deeponet --branch_layers 3 128 128 128 4 --trunk_layers 1 128 128 2 --epochs 250 --load_data False --multi_output_strategy orthonormal_split_branch --num_outputs 2 --experiment_name "$SCRIPT_NAME"

#wider net

#            --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/ex42_1/exp_n_20250325-204226/models/best_ckpt_epoch_475.pth \


counter=1

for detach in False True; do
    python main.py --problem harmonic_oscillator --device cpu --method deeponet \
            --branch_layers 3 128 128 128 16 --trunk_layers 1 128 128 128 8 \
            --tmax 10 \
            --use_implicit_nrg True \
            --num_norm_refinements 1 \
            --epochs 500 --load_data False --loss mse \
            --strategy normal \
            --branch_weight 0 --trunk_weight 0 \
            --num_outputs 2 --experiment_name "${SCRIPT_NAME}_${counter}"
        ((counter++))
done
