SCRIPT_NAME=$(basename "$0" .sh)

# Run deeponet on harmonic oscillator problem
# python main.py --problem harmonic_oscillator --device cpu --method deeponet --branch_layers 3 128 128 128 4 --trunk_layers 1 128 128 2 --epochs 250 --load_data False --multi_output_strategy orthonormal_split_branch --num_outputs 2 --experiment_name "$SCRIPT_NAME"

#wider net

counter=1


#num_ouputs is M+1 (number of fourier modes) (?)
#then last branch layer should be 2K*(M+1)
#and last trunk layer should be K 

# --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/ex31_1/exp_n_20250317-143256/models/best_ckpt_epoch_865.pth \

# Fourier FourierQR

for strat in Fourier; do
    python main.py --problem 1d_wave --device cpu --method deeponet \
            --eval_only True \
            --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/dos_1_1/exp_n_20250419-002159/models/best_ckpt_epoch_35.pth \
            --branch_layers 2 128 128 128 160 --trunk_layers 1 128 128 128 64 \
            --IC '{"c": 10, "type": "periodic_gp", "params": {"lengthscale":0.1, "variance":1.0}}' \
            --fourier_input True \
            --num_input_fn 2 \
            --num_output_fn 2 \
            --x_filter_cutoff_ratio 0.1 \
            --Nx 199 \
            --Nt 199 \
            --x_res 1.282 \
            --t_res 0.05 \
            --data_dt 0.0001 \
            --data_modes 10 \
            --zero_zero_mode True \
            --tmin 0 \
            --tmax 2 \
            --lr 1e-3 \
            --use_ifft True \
            --epochs 1000 \
            --n_branch 500 \
            --loss mse \
            --track_all_losses 0 \
            --strategy $strat \
            --branch_weight 0 --trunk_weight 0 \
            --experiment_name "${SCRIPT_NAME}_${counter}"
        ((counter++))
done

for strat in FourierNorm; do
    python main.py --problem 1d_wave --device cpu --method deeponet \
            --eval_only True \
            --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/dos_1_2/exp_n_20250419-030258/models/best_ckpt_epoch_75.pth \
            --branch_layers 2 128 128 128 160 --trunk_layers 1 128 128 128 64 \
            --IC '{"c": 10, "type": "periodic_gp", "params": {"lengthscale":0.1, "variance":1.0}}' \
            --fourier_input True \
            --num_input_fn 2 \
            --num_output_fn 2 \
            --x_filter_cutoff_ratio 0.1 \
            --Nx 199 \
            --Nt 199 \
            --x_res 1.282 \
            --t_res 0.05 \
            --data_dt 0.0001 \
            --data_modes 10 \
            --zero_zero_mode True \
            --tmin 0 \
            --tmax 2 \
            --lr 1e-3 \
            --use_ifft True \
            --epochs 1000 \
            --n_branch 500 \
            --loss mse \
            --track_all_losses 0 \
            --strategy $strat \
            --branch_weight 0 --trunk_weight 0 \
            --experiment_name "${SCRIPT_NAME}_${counter}"
        ((counter++))
done

for strat in FourierQR; do
    python main.py --problem 1d_wave --device cpu --method deeponet \
            --eval_only True \
            --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/dos_1_3/exp_n_20250419-060358/models/best_ckpt_epoch_70.pth \
            --branch_layers 2 128 128 128 160 --trunk_layers 1 128 128 128 64 \
            --IC '{"c": 10, "type": "periodic_gp", "params": {"lengthscale":0.1, "variance":1.0}}' \
            --fourier_input True \
            --num_input_fn 2 \
            --num_output_fn 2 \
            --x_filter_cutoff_ratio 0.1 \
            --Nx 199 \
            --Nt 199 \
            --x_res 1.282 \
            --t_res 0.05 \
            --data_dt 0.0001 \
            --data_modes 10 \
            --zero_zero_mode True \
            --tmin 0 \
            --tmax 2 \
            --lr 1e-3 \
            --use_ifft True \
            --epochs 10 \
            --n_branch 500 \
            --loss mse \
            --track_all_losses 0 \
            --strategy $strat \
            --branch_weight 0 --trunk_weight 0 \
            --experiment_name "${SCRIPT_NAME}_${counter}"
        ((counter++))
done

