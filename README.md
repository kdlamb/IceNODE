# IceNODE
This github repo provides the analysis and code for a scientific machine learning analysis of depositional ice growth models. 
Here we explore how neural ordinary differential equations and equation discovery can be applied to observations of times series of the mass of single ice crystals grown in a levitation diffusion chamber.

## Overview
This repo contains a PyTorch implementation of the code for the paper "Discovering How Ice Crystals Grow Using Neural Ordinary Differential Equations and Symbolic Regression"

## Content
- [Data Preparation](#data-preparation)
- [NODE Model](#node-model)
- [Symbolic Regression](#symbolic-regression)


## Data Preparation
Data sets for the levitation diffusion chamber experiments are described in Pokrifka et al. 2020 and Pokrifka et al. 2023. 
The preprocessing.py script creates a pytorch data loader containing all experiments used in this analysis. 

Synthetic data for the levitation diffusion chamber experiments assumes a functional form the depositional ice growth model, and the script synthetic_data.py 
takes the initial conditions and detrended noise from the real experiments to initiate the synthetic data sets.

Data sets for the AIDA cloud chamber experiment are described in Lamb et al. 2017, Clouser et al. 2020, and Lamb et al. 2023.

## NODE Models
The main.py script is used for training the NODE model against the experimental observations or synthetic data sets. It includes weak, medium, and strong assumptions for the physical constraints
used in developing the NODE models.

## Symbolic Regression
The [PySR library](https://github.com/MilesCranmer/PySR) is used for symbolic regression. 

## Citation
The preprint for this paper can be found at:
```
@article{Lamb2025,
  title={Discovering How Ice Crystals Grow Using Neural Ordinary Differential Equations and Symbolic Regression},
  author={Lamb, K.D. and J. Harrington},
  journal={ESS Open Archive},
  doi = {DOI: },
  year={2025}
}
```