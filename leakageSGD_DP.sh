#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#FEDSGD

# WEAR
python Leakage.py --config ./configs/leakage/wear_loso_deep.yaml --seed 1 --eval_type loso --trained True --neptune True --attack _default_optimization_attack --sampling balanced
python Leakage.py --config ./configs/leakage/wear_loso_tiny.yaml --seed 1 --eval_type loso --trained True --neptune True --attack _default_optimization_attack --sampling balanced
python Leakage.py --config ./configs/leakage/wear_loso_deep.yaml --seed 1 --eval_type loso --trained True --neptune True --attack _default_optimization_attack --sampling unbalanced
python Leakage.py --config ./configs/leakage/wear_loso_tiny.yaml --seed 1 --eval_type loso --trained True --neptune True --attack _default_optimization_attack --sampling unbalanced

# WETLAB
python Leakage.py --config ./configs/leakage/wetlab_loso_deep.yaml --seed 1 --eval_type loso --trained True --neptune True --attack _default_optimization_attack --sampling balanced
python Leakage.py --config ./configs/leakage/wetlab_loso_tiny.yaml --seed 1 --eval_type loso --trained True --neptune True --attack _default_optimization_attack --sampling balanced
python Leakage.py --config ./configs/leakage/wetlab_loso_deep.yaml --seed 1 --eval_type loso --trained True --neptune True --attack _default_optimization_attack --sampling unbalanced
python Leakage.py --config ./configs/leakage/wetlab_loso_tiny.yaml --seed 1 --eval_type loso --trained True --neptune True --attack _default_optimization_attack --sampling unbalanced