
# DiffMeta: Inverse Design of Metamaterials with Manufacturing-Aware Spectrum-to-Shape Diffusion Models

This repository implements the paper [Inverse Design of Metamaterials with Manufacturing-Aware Spectrum-to-Shape Diffusion Models](LINK).

## Introduction
### 1. The task description for inverse design of metamaterial:
![Task illustration](./figures/metamaterial_design.png)
As illustrated in the above Figure, the MIM (Metal-Insulator-Metal) tri-layer metamaterial periodic structure is used in this work, consisting of a free-form gold (Au) top layer, an amorphous silicon (Si) insulator thin film middle layer, and a uniform gold (Au) thin film bottom layer. Given a desired spectrum, the inverse design task involves generating the optimal pattern and parameters for a 64×64 pixel periodic unit cell with 4-dimensional size composition parameters. The design parameters include pitch size (ϕ_1), height of the top shape pattern (ϕ_2), middle dielectric spacer size (ϕ_3), and bottom reflector size (ϕ_4). 

### 2. DiffMeta
![Model illustrations](./figures/framework.png)

## Data
The data used in this paper, including generated and training data, is publicly available on Zenodo. You can access it here: https://zenodo.org/records/12797962.

## Dependencies
### Conda environment
Please use our environment file to set up the required environment:.
'''
# Clone the environment
conda env create -f environment.yml
# Activate the environment
conda activate diffmeta
'''
### S4 Installation
For spectral simulation, install S4 according to the [guidance](https://web.stanford.edu/group/fan/S4/install.html)

## Usage
### 1. Training
First, train a spectral autoencoder:
'''
python train_spec_encoder.py
'''
Then train DiffMeta using the following command:
'''
python train_diffmeta.py
'''

### 2. Inference
To perform inference with a trained DiffMeta model, use:
### 3. Generate metamaterial structures for given spectra
To generate metamaterial structures based on specific spectra, run:
### 4. Simulation
To simulate the optical properties of the generated metamaterials, use:

## Benchmark

## Cite

If you reference or cite this work in your research, please cite:
.......
