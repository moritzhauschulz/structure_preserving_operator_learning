SCRIPT_NAME=$(basename "$0" .sh)

# Run deeponet on harmonic oscillator problem
# python main.py --problem harmonic_oscillator --device cpu --method deeponet --branch_layers 3 128 128 128 4 --trunk_layers 1 128 128 2 --epochs 250 --load_data False --multi_output_strategy orthonormal_split_branch --num_outputs 2 --experiment_name "$SCRIPT_NAME"

#wider net

counter=1


#num_ouputs is M+1 (number of fourier modes) (?)
#then last branch layer should be 2K*(M+1)
#and last trunk layer should be K 

for strat in FourierNorm; do
    python main.py --problem 1d_KdV_Soliton --device cpu --method deeponet \
            --branch_layers 2 128 128 128 160 --trunk_layers 1 128 128 128 4 \
            --num_outputs 20 \
            --tmax 1 \
            --x_res 0.05 \
            --t_res 0.1 \
            --epochs 5 --load_data False --loss mse \
            --track_all_losses 0 \
            --multi_output_strategy $strat \
            --branch_weight 0 --trunk_weight 0 \
            --experiment_name "${SCRIPT_NAME}_${counter}" \
            --IC '{"c": [1,3], "a": [-2,2]}'
        ((counter++))
done
