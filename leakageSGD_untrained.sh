#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#FEDSGD

# WEAR
python Leakage.py --config ./configs/leakage/wear_loso_deep.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling balanced --batch_size 1
python Leakage.py --config ./configs/leakage/wear_loso_tiny.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling unbalanced --batch_size 1

python Leakage.py --config ./configs/leakage/wear_loso_deep.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling balanced --batch_size 10
python Leakage.py --config ./configs/leakage/wear_loso_tiny.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling unbalanced --batch_size 10

python Leakage.py --config ./configs/leakage/wear_loso_deep.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling balanced --batch_size 100
python Leakage.py --config ./configs/leakage/wear_loso_tiny.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling unbalanced --batch_size 100

# WETLAB
python Leakage.py --config ./configs/leakage/wetlab_loso_deep.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling balanced --batch_size 1
python Leakage.py --config ./configs/leakage/wetlab_loso_tiny.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling unbalanced --batch_size 10

python Leakage.py --config ./configs/leakage/wetlab_loso_deep.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack--sampling balanced --batch_size 10
python Leakage.py --config ./configs/leakage/wetlab_loso_tiny.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack--sampling unbalanced --batch_size 10

python Leakage.py --config ./configs/leakage/wetlab_loso_deep.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling balanced --batch_size 100 
python Leakage.py --config ./configs/leakage/wetlab_loso_tiny.yaml --eval_type loso --trained False --neptune True --attack _default_optimization_attack --sampling unbalanced --batch_size 100 