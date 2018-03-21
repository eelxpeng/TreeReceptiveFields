#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 3 --name agnews-treeconv-layer-3 > logs/log-agnews-treeconv-layer-3-run-1