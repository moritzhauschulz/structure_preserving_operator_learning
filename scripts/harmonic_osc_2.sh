SCRIPT_NAME=$(basename "$0" .sh)

# Run deeponet on harmonic oscillator problem
# python main.py --problem harmonic_oscillator --device cpu --method deeponet --branch_layers 3 128 128 128 4 --trunk_layers 1 128 128 2 --epochs 250 --load_data False --multi_output_strategy orthonormal_split_branch --num_outputs 2 --experiment_name "$SCRIPT_NAME"

#wider net

counter=4

for strat in vanilla QR normal; do
    for i in {1}; do
        python main.py --problem harmonic_oscillator --device cpu --method deeponet \
                --branch_layers 3 128 128 128 8 --trunk_layers 1 128 128 128 4 \
                --tmax 10 \
                --epochs 500 --load_data False --loss mse \
                --strategy $strat \
                --branch_weight 0 --trunk_weight 0 \
                --num_outputs 2 --experiment_name "${SCRIPT_NAME}_${counter}" \
                --IC '{"q0": [-1,1], "p0": [-1,1], "omega": [1,1]}'
    done
    ((counter++))   
done

for strat in normal; do
    for i in {1}; do
        python main.py --problem harmonic_oscillator --device cpu --method deeponet \
                --branch_layers 3 128 128 128 8 --trunk_layers 1 128 128 128 4 \
                --tmax 10 \
                --use_implicit_nrg True \
                --epochs 500 --load_data False --loss mse \
                --strategy $strat \
                --branch_weight 0 --trunk_weight 0 \
                --num_outputs 2 --experiment_name "${SCRIPT_NAME}_${counter}" \
                --IC '{"q0": [-1,1], "p0": [-1,1], "omega": [1,1]}'
        done
    ((counter++))   
done
