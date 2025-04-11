SCRIPT_NAME=$(basename "$0" .sh)

# Run deeponet on harmonic oscillator problem
# python main.py --problem harmonic_oscillator --device cpu --method deeponet --branch_layers 3 128 128 128 4 --trunk_layers 1 128 128 2 --epochs 250 --load_data False --multi_output_strategy orthonormal_split_branch --num_outputs 2 --experiment_name "$SCRIPT_NAME"

#wider net

counter=1


#num_ouputs is M+1 (number of fourier modes) (?)
#then last branch layer should be 2K*(M+1)
#and last trunk layer should be K 

# --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/ex31_1/exp_n_20250317-143256/models/best_ckpt_epoch_865.pth \


for strat in FullFourierNorm; do
    for iter in {1,2,3}; do
        python main.py --problem 1d_wave --device cpu --method full_fourier \
                --branch_layers 2 128 128 128 160 --trunk_layers 1 128 128 128 4 \
                --IC '{"c": 10, "type": "periodic_gp", "params": {"lengthscale":0.1, "variance":1.0}}' \
                --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/full_fourier/experiments/afno_3_1/exp_n_20250410-224916/models/best_ckpt_epoch_100.pth \
                --fourier_input True \
                --num_input_fn 2 \
                --num_output_fn 1 \
                --num_norm_refinements $iter \
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
                --epochs 50 \
                --n_branch 100 \
                --loss mse \
                --track_all_losses 0 \
                --strategy $strat \
                --branch_weight 0 --trunk_weight 0 \
                --experiment_name "${SCRIPT_NAME}_${counter}"
            ((counter++))
    done
done

