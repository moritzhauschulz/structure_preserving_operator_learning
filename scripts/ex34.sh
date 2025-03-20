SCRIPT_NAME=$(basename "$0" .sh)

# Run deeponet on harmonic oscillator problem
# python main.py --problem harmonic_oscillator --device cpu --method deeponet --branch_layers 3 128 128 128 4 --trunk_layers 1 128 128 2 --epochs 250 --load_data False --multi_output_strategy orthonormal_split_branch --num_outputs 2 --experiment_name "$SCRIPT_NAME"

#wider net

counter=1


#num_ouputs is M+1 (number of fourier modes) (?)
#then last branch layer should be 2K*(M+1)
#and last trunk layer should be K 

# --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/ex31_1/exp_n_20250317-143256/models/best_ckpt_epoch_865.pth \


for strat in FourierNorm Fourier; do
    python main.py --problem 1d_wave --device cpu --method deeponet \
            --branch_layers 2 128 128 128 160 --trunk_layers 1 128 128 128 4 \
            --IC '{"c": 5, "type": "periodic_gp", "params": {"lengthscale":0.1, "variance":1.0}}' \
            --fourier_input True \
            --num_input_fn 2 \
            --num_output_fn 1 \
            --col_N 199 \
            --x_res 1.282 \
            --t_res 0.05 \
            --data_dt 0.001 \
            --data_modes 10 \
            --zero_zero_mode True \
            --tmin 0 \
            --tmax 3 \
            --lr 1e-3 \
            --use_ifft True \
            --epochs 1000 \
            --n_branch 100 \
            --loss mse \
            --track_all_losses 0 \
            --multi_output_strategy $strat \
            --branch_weight 0 --trunk_weight 0 \
            --experiment_name "${SCRIPT_NAME}_${counter}"
        ((counter++))
done
