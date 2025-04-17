SCRIPT_NAME=$(basename "$0" .sh)

# Run deeponet on harmonic oscillator problem
# python main.py --problem harmonic_oscillator --device cpu --method deeponet --branch_layers 3 128 128 128 4 --trunk_layers 1 128 128 2 --epochs 250 --load_data False --multi_output_strategy orthonormal_split_branch --num_outputs 2 --experiment_name "$SCRIPT_NAME"

#wider net

counter=1

normal implicit energy
for strat in vanilla; do
    python main.py --problem harmonic_oscillator --device cpu --method deeponet \
        --eval_only True \
        --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/harmonic_osc_1_1/exp_n_20250410-115608/models/best_ckpt_epoch_475.pth \
        --branch_layers 3 128 128 128 8 --trunk_layers 1 128 128 128 4 \
        --tmax 10 \
        --epochs 10 --load_data False --loss mse \
        --strategy $strat \
        --branch_weight 0 --trunk_weight 0 \
        --num_outputs 2 --experiment_name "${SCRIPT_NAME}_${counter}"
done
((counter++))   


normal learned energy
for strat in normal; do
    python main.py --problem harmonic_oscillator --device cpu --method deeponet \
        --eval_only True \
        --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/harmonic_osc_1_2/exp_n_20250410-133226/models/best_ckpt_epoch_480.pth \
        --branch_layers 3 128 128 128 8 --trunk_layers 1 128 128 128 4 \
        --tmax 10 \
        --epochs 10 --load_data False --loss mse \
        --strategy $strat \
        --branch_weight 0 --trunk_weight 0 \
        --num_outputs 2 --experiment_name "${SCRIPT_NAME}_${counter}"
done
((counter++))   


#normal implicit energy
for strat in QR; do
    python main.py --problem harmonic_oscillator --device cpu --method deeponet \
        --eval_only True \
        --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/harmonic_osc_1_3/exp_n_20250410-141041/models/best_ckpt_epoch_295.pth \
        --branch_layers 3 128 128 128 8 --trunk_layers 1 128 128 128 4 \
        --tmax 10 \
        --epochs 10 --load_data False --loss mse \
        --strategy $strat \
        --branch_weight 0 --trunk_weight 0 \
        --num_outputs 2 --experiment_name "${SCRIPT_NAME}_${counter}"
done
((counter++))   


normal implicit energy
for strat in normal; do
    python main.py --problem harmonic_oscillator --device cpu --method deeponet \
        --eval_only True \
        --load_checkpoint /Users/moritzhauschulz/oxford_code/structure_preserving_operator_learning/methods/deeponet/experiments/harmonic_osc_2_1/exp_n_20250416-154223/models/best_ckpt_epoch_485.pth \
        --branch_layers 3 128 128 128 8 --trunk_layers 1 128 128 128 4 \
        --tmax 10 \
        --use_implicit_nrg True \
        --epochs 500 --load_data False --loss mse \
        --strategy $strat \
        --branch_weight 0 --trunk_weight 0 \
        --num_outputs 2 --experiment_name "${SCRIPT_NAME}_${counter}"
done
((counter++))   


