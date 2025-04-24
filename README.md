# Structure Preserving Operator Learning

## Abstract

## Author

- **Moritz Elias Hauschulz** - University of Oxford - moritz.hauschulz@stx.ox.ac.uk

## Supervisors

- **Georg Maierhofer** - University of Oxford
- **Nicolas Boullé** - Imperial College London


## Methods
- my methods:
    - Adaptations of DeepONet (for harmonic oscillator)
      - QR DeepONet
      - Normalised DeepONet
      - Implicitly Normalised DeepONet
    - DeepONet Spectral Operator (DSO) (for wave equation)
      - QR DSO
      - Normalised DSO
      - Implicitly Normalised DSO
      - Vanilla
    - Augmented Fourier Neural Operator (AFNO) (for wave equation)
      - Normalised
      - Vanilla

## Installation and Example (contact moritz.hauschulz@stx.ox.ac.uk for queries)

To reproduce the results or use the model, follow these steps:

1. Clone the repository:
    ```bash
    git clone [https://github.com/moritzhauschulz/samplingEBMs.git](https://github.com/moritzhauschulz/structure_preserving_operator_learning.git)
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Using WANDB
    If you do not wish to use WANDB for tracking training dynamics, choose --wandb False. Else, specify --wandb_user and --wandb_project.

5. Run codes for experiments (NOTE: adapting the specifications requires some insight into the code structure, especially of main.py):
    ```bash 
    bash scripts/afno_1.sh
    ```
    ```bash
    bash scripts/dos_1.sh
    ```
    ```bash
    bash scripts/harmonic_osc_1.sh
    ```
    - Output will appear in nested folders under the respective method for each run.
    - The skeleton codes in
      ```
      wave_plots.ipynb
      ```
      and
      ```
      harmonic_osc_plots.ipynb
      ```
      can be used to produce plots, but this requires adaptation and must be used toegether with WANDB.

