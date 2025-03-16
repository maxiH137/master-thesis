# Differential Privacy for Human Activity Recognition

This repository contains the code for the master thesis "Differential Privacy for Human Activity Recognition" of Maximilian Hopp.
The thesis is submitted to Professor Michael Moeller at the University of Siegen.

## Introduction
Gradient Inversion (GI) is a problem that emerged after the develop-
ment of the Federated Learning (FL) algorithm. In FL a neural net-
work is jointly trained through multiple devices by sharing gradients
with a central server [24]. From these gradients, the original model in-
put can be reconstructed in certain cases. In this thesis, the focus in GI
is shifted to Human Activity Recognition (HAR) with focus on a sensor
setting, as research in this field focused mainly on image datasets and
reconstructing the original image and its label. We adapt this for the
two HAR datasets WEAR [4] and Wetlab [37]. We evaluated the perfor-
mance of existing label reconstruction attacks for different settings of
the FL-algorithm, model, training stage, batch size, and sampling strat-
egy. The effectiveness of the commonly used defense strategy differen-
tial privacy, implemented through gradient clipping and noise addition,
is also tested for a range of settings. The main focus is on the FedSGD
algorithm, but preliminary results for the FedAVG algorithm are also
provided. The results indicate that label reconstruction in HAR sensing
is strongly influenced by the number of classes, sampling strategy, and
data distribution in the dataset. The latter seems also to influence the
necessary magnitude of clipping and noise to make the gradients ro-
bust against attacks. Individual subjects can have a difference of 10%
in the accuracy of label reconstruction due to the different distribution
of their data.

(Abstract from the "Master Thesis Maximilian Hopp.pdf")

## Requirements
Use requirements.txt to create necessary conda environment.
It is basically the requirements of the TAL repository including an import of the breaching repository.


## Experiments
The main function for FedSGD evaluation is Leakage.py. 
It is necessary to include the dataset, train the models from the TAL repository and integrate the breaching repository before starting experiments.

The main functions for FedAVG evaluation are LeakageAVG_local.py and LeakageAVG_mult.py.
It is necessary to include the dataset, train the models from the TAL repository and integrate the breaching repository before starting experiments.
In these scripts config files of the breaching repository are used and need to be adjusted accordingly. 

# Evaluation
In the eval folder different scripts can be found to load runs from neptune, group them and average them. 
The most up to date ones are Eval_all.py and the genPlots python scripts. 
Eval_all creates an output file that is loaded by genPlots, which creates Plots and grouped/averaged .csv files for use in Excel. 

## Ackownledgement
For the model training the TAL repository by Marius Bock is used as well as the WEAR dataset (https://github.com/mariusbock/tal_for_har).
The label reconstruction attacks are evaluated with the breaching repository by Geiping et al. (https://github.com/JonasGeiping/breaching).
Additional attacks are added from Ma et al. (https://github.com/BUAA-CST/iLRG) and Gat et al. (https://github. com/nadbag98/LLBG). 
