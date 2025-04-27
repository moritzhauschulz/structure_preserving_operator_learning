SCRIPT_NAME=$(basename "$0" .sh)

# Run AFNO on wave equation – data generation may take significant time on first run

counter=1


for strat in FullFourier; do
    python main.py --problem 1d_wave --device cpu --method full_fourier \
            --branch_layers 2 128 128 128 160 --trunk_layers 1 128 128 128 4 \
            --IC '{"c": 10, "type": "periodic_gp", "params": {"lengthscale":0.1, "variance":1.0}}' \
            --fourier_input True \
            --num_input_fn 2 \
            --num_output_fn 1 \
            --Nx 199 \
            --Nt 199 \
            --x_res 1.282 \
            --t_res 0.05 \
            --data_dt 0.0001 \
            --data_modes 10 \
            --zero_zero_mode True \
            --t_filter_cutoff_ratio 0.1 \
            --x_filter_cutoff_ratio 0.1 \
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

for strat in FullFourierNorm; do
    python main.py --problem 1d_wave --device cpu --method full_fourier \
            --branch_layers 2 128 128 128 160 --trunk_layers 1 128 128 128 4 \
            --IC '{"c": 10, "type": "periodic_gp", "params": {"lengthscale":0.1, "variance":1.0}}' \
            --fourier_input True \
            --num_input_fn 2 \
            --num_output_fn 1 \
            --Nx 199 \
            --Nt 199 \
            --x_res 1.282 \
            --t_res 0.05 \
            --data_dt 0.0001 \
            --data_modes 10 \
            --zero_zero_mode True \
            --t_filter_cutoff_ratio 0.1 \
            --x_filter_cutoff_ratio 0.1 \
            --tmin 0 \
            --tmax 2 \
            --lr 0.1 \
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

