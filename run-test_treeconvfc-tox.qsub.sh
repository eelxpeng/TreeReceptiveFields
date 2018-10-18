#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e
#$ -l h=client110

CUDA_VISIBLE_DEVICES=1 python test_treeconvfc-tox.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --name tox-treeconvfc-layer-3 |& tee logs/log-tox-treeconvfc-layer-3-run-4
