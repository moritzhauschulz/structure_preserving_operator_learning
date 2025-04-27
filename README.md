# Towards Strict Preservation of Integral Quantities in Operator Learning

## Abstract
Machine learning methods have become increasingly capable of modeling dynamical systems governed by ordinary and partial differential equations. However, ensuring that model outputs exhibit realistic physical behaviour beyond the training set remains a challenge. This paper is dedicated to developing architectures to strictly enforce preservation of integral quantities which are known to be conserved by the underlying system on which the models are trained. To motivate the approach, we propose two schemes for enforcing energy conservation in Deep Operator Networks (DeepONets) and show their effectiveness on the example of the harmonic oscillator. We then transfer these approaches to the spectral domain, where besides general theory we consider the special case of the wave equation. This leads us to propose not only a new class of models which we term DeepONet Spectral Operators (DSO), but also a versatile framework for conserving general integral quantities satisfied by a large class of linear homogeneous partial differential equations (PDE). We show that approximate energy conservation can be achieved with little computational overhead, but also provide an argument for why exact energy preservation through the proposed methods suffers from errors due to non-linear feedback. We further present a fully spectral alternative, which is related to the Fourier Neural Operator (FNO). We show empirically that this Augmented Fourier Neural Operator (AFNO) can be forced to conserve energy exactly, besides outperforming the DSO by an order of magnitude.

## Author 

- **ANONYMIZED FOR GRADING**

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

## Installation and Example (contact ANONIMIZED for queries)

To reproduce the results or use the model, follow these steps:

1. Clone the repository:
    ```bash
    git clone [https://github.com/moritzhauschulz/samplingEBMs.git](https://github.com/moritzhauschulz/structure_preserving_operator_learning.git)
    ```
2. Make virtual environment
    ```bash
    python -m venv .venv
    ```
    ```bash
    source .venv/bin/activate
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
    bash scripts/dso_1.sh
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
      can be used to produce plots, but this requires adaptation and must be used together with WANDB.

