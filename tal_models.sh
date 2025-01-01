#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

python main.py --config ./configs/baseline/deepconvlstm/wear_loso.yaml --seed 1

python main.py --config ./configs/baseline/deepconvlstm/wetlab_loso.yaml --seed 1  

python main.py --config ./configs/baseline/tinyhar/wear_loso.yaml --seed 1 

python main.py --config ./configs/baseline/tinyhar/wetlab_loso.yaml --seed 1 