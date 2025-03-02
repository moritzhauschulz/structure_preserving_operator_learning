SCRIPT_NAME=$(basename "$0" .sh)

# Run deeponet on harmonic oscillator problem
python main.py --problem harmonic_oscillator --device cpu --method deeponet --epochs 10 --load_data False --experiment_name "$SCRIPT_NAME"
