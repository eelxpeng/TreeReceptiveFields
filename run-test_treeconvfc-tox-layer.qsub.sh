#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 3 --name tox-treeconvfc-layer-3 > logs/log-tox-treeconvfc-layer-3-run-1
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 1 --name tox-treeconvfc-layer-1 > logs/log-tox-treeconvfc-layer-1-run-1
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 2 --name tox-treeconvfc-layer-2 > logs/log-tox-treeconvfc-layer-2-run-1
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 4 --name tox-treeconvfc-layer-4 > logs/log-tox-treeconvfc-layer-4-run-1
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 5 --name tox-treeconvfc-layer-5 > logs/log-tox-treeconvfc-layer-5-run-1
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 6 --name tox-treeconvfc-layer-6 > logs/log-tox-treeconvfc-layer-6-run-1
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 7 --name tox-treeconvfc-layer-7 > logs/log-tox-treeconvfc-layer-7-run-1

python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 3 --name tox-treeconvfc-layer-3 > logs/log-tox-treeconvfc-layer-3-run-2
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 1 --name tox-treeconvfc-layer-1 > logs/log-tox-treeconvfc-layer-1-run-2
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 2 --name tox-treeconvfc-layer-2 > logs/log-tox-treeconvfc-layer-2-run-2
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 4 --name tox-treeconvfc-layer-4 > logs/log-tox-treeconvfc-layer-4-run-2
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 5 --name tox-treeconvfc-layer-5 > logs/log-tox-treeconvfc-layer-5-run-2
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 6 --name tox-treeconvfc-layer-6 > logs/log-tox-treeconvfc-layer-6-run-2
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 7 --name tox-treeconvfc-layer-7 > logs/log-tox-treeconvfc-layer-7-run-2

python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 3 --name tox-treeconvfc-layer-3 > logs/log-tox-treeconvfc-layer-3-run-3
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 1 --name tox-treeconvfc-layer-1 > logs/log-tox-treeconvfc-layer-1-run-3
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 2 --name tox-treeconvfc-layer-2 > logs/log-tox-treeconvfc-layer-2-run-3
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 4 --name tox-treeconvfc-layer-4 > logs/log-tox-treeconvfc-layer-4-run-3
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 5 --name tox-treeconvfc-layer-5 > logs/log-tox-treeconvfc-layer-5-run-3
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 6 --name tox-treeconvfc-layer-6 > logs/log-tox-treeconvfc-layer-6-run-3
python test_treeconvfc-tox-layer.py --batch-size 256 --epochs 50 --dataset tox --lr 0.001 --num-layer 7 --name tox-treeconvfc-layer-7 > logs/log-tox-treeconvfc-layer-7-run-3
