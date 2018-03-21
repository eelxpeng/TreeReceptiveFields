#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

python test_mnist.py --epochs 50 --lr 0.01