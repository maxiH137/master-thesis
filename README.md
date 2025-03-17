# Differential Privacy for Human Activity Recognition

This repository contains the code for the master thesis "Differential Privacy for Human Activity Recognition" of Maximilian Hopp.
The thesis is submitted to Professor Michael Möller at the University of Siegen.

## Introduction
Gradient Inversion (GI) is a problem that emerged after the develop-
ment of the Federated Learning (FL) algorithm. In FL a neural net-
work is jointly trained through multiple devices by sharing gradients
with a central server [24]. From these gradients, the input of the orig-
inal model can be reconstructed in certain cases. In this thesis, the
focus in GI is shifted to Human Activity Recognition (HAR) with focus
on a sensor setting, as research in this field mainly used image datasets
where the original image and label were reconstructed. [14, 41] We
evaluated the performance of existing label reconstruction attacks for
different settings of the FL-algorithm, model, training stage, batch size,
and sampling strategy. The two HAR datasets WEAR [4] and Wetlab
[37] are used for sensor input. The effectiveness of the commonly used
defense strategy differential privacy [27] is implemented through gra-
dient clipping and noise addition to the gradients. The main focus is
on the FedSGD algorithm, but preliminary results for the FedAVG al-
gorithm are also provided. The findings suggest that in HAR sensing,
label reconstruction is significantly affected by the number of classes,
sampling method, and distribution of the classes in a dataset. In par-
ticular, there are substantial differences in leakage between the WEAR
and Wetlab datasets. The latter seems also to influence the necessary
magnitude of clipping and noise to make the gradients robust against
attacks. Individual subjects showed a difference of up to 10% in the ac-
curacy of label reconstruction, possibly due to the different distribution
of their data.

(Abstract from "Master_Thesis Maximilian Hopp.pdf")

## Requirements
Use requirements.txt to create necessary conda environment.
It is basically the requirements of the TAL repository including an import of the breaching repository.
Some files in the breaching repository need to be adjusted, to fit the HAR setting.
In the folder breaching all files are contained, even unchanged ones. 
The most important changes are in base_attack.py, where the label reconstruction attacks are executed.

Setup:
1. Clone repository
2. Create TAL Conda environment or use a fresh Conda env and run pip install -r requirements.txt
3. Install breaching (if not installed by requirements.txt)
4. Change breaching files with the breaching files from this repository
5. Set up WEAR and Wetlab dataset
6. Create checkpoints of DeepConvLSTM and TinyHAR with the TAL repository. Include them in folder "saved_models".


## Experiments
The main function for FedSGD evaluation is Leakage.py. 
It is necessary to include the dataset, train the models from the TAL repository and integrate the breaching repository before starting experiments.

The main functions for FedAVG evaluation are LeakageAVG_local.py and LeakageAVG_mult.py.
It is necessary to include the dataset, train the models from the TAL repository and integrate the breaching repository before starting experiments.
In these scripts config files of the breaching repository are used and need to be adjusted accordingly. 
For example the noise and clipping can be adjusted in the breaching repository config.

The command to execute the scripts are:
- <p>Leakage.py --./configs/leakage/wear_loso_deep.yaml --batch_size 100 --trained </p>
- <p>LeakageAVG_local.py --./configs/leakage/wear_loso_deep.yaml --batch_size 100 --trained True --num_data_points 500 --num_data_per_local_update_step 100 --num_local_updates 5</p>
- <p>LeakageAVG_mult.py --./configs/leakage/wear_loso_deep.yaml --trained True --num_data_points 500 --num_data_per_local_update_step 100 --num_local_updates 5 --user_range 5</p>

## Evaluation
In the eval folder different scripts can be found to load runs from neptune, group them and average them. 
The most up to date ones are Eval_all.py and the genPlots python scripts. 
Eval_all creates an output file that is loaded by genPlots, which creates Plots and grouped/averaged .csv files for use in Excel. 

## Ackownledgement
For the model training the TAL repository by Marius Bock is used as well as the WEAR and Wetlab dataset implementation (https://github.com/mariusbock/tal_for_har).
The label reconstruction attacks are evaluated with the breaching repository by Geiping et al. (https://github.com/JonasGeiping/breaching).
Additional attacks are added from Ma et al. (https://github.com/BUAA-CST/iLRG) and Gat et al. (https://github. com/nadbag98/LLBG). 
