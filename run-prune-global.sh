#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

# echo "Experiment with Tox21 with layer-wise pruning" 1>&2
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 0 --num-layer 4 --shape 1 --num-neuron 512 --sparsity 0.37 --globalprune --name globalprune-tox-0-l4-con-512
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 1 --num-layer 1 --shape 0 --num-neuron 1024 --sparsity 0.2 --globalprune --name globalprune-tox-1-l1-rec-1024
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 2 --num-layer 1 --shape 0 --num-neuron 1024 --sparsity 0.2 --globalprune --name globalprune-tox-2-l1-rec-1024
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 3 --num-layer 1 --shape 0 --num-neuron 512 --sparsity 0.4 --globalprune --name globalprune-tox-3-l1-rec-512
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 4 --num-layer 2 --shape 0 --num-neuron 1024 --sparsity 0.12 --globalprune --name globalprune-tox-4-l2-rec-1024
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 5 --num-layer 4 --shape 0 --num-neuron 512 --sparsity 0.21 --globalprune --name globalprune-tox-5-l4-rec-512
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 6 --num-layer 2 --shape 1 --num-neuron 512 --sparsity 0.39 --globalprune --name globalprune-tox-6-l2-con-512
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 7 --num-layer 4 --shape 0 --num-neuron 1024 --sparsity 0.07 --globalprune --name globalprune-tox-7-l4-rec-1024
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 8 --num-layer 1 --shape 0 --num-neuron 512 --sparsity 0.4 --globalprune --name globalprune-tox-8-l1-rec-512
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 9 --num-layer 2 --shape 0 --num-neuron 512 --sparsity 0.3 --globalprune --name globalprune-tox-9-l2-rec-512
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 10 --num-layer 1 --shape 0 --num-neuron 1024 --sparsity 0.2 --globalprune --name globalprune-tox-10-l1-rec-1024
# python prune.py --batch-size 100 --epochs 300 --dataset tox --target 11 --num-layer 1 --shape 0 --num-neuron 512 --sparsity 0.4 --globalprune --name globalprune-tox-11-l1-rec-512


# echo "Experiment with agnews with layer-wise pruning" 1>&2
# python prune.py --batch-size 1000 --epochs 300 --dataset agnews --num-layer 3 --shape 0 --num-neuron 2048 --sparsity 0.06 --globalprune --name globalprune-agnews-l3-rec-2048
# python prune.py --batch-size 1000 --epochs 300 --dataset dbpedia --num-layer 2 --shape 1 --num-neuron 1024 --sparsity 0.17 --globalprune --name globalprune-dbpedia-l2-con-1024
# python prune.py --batch-size 1000 --epochs 300 --dataset sogounews --num-layer 4 --shape 0 --num-neuron 1024 --sparsity 0.14 --globalprune --name globalprune-sogounews-l4-rec-1024
# python prune.py --batch-size 1000 --epochs 300 --dataset yelp --num-layer 2 --shape 0 --num-neuron 512 --sparsity 0.32 --globalprune --name globalprune-yelp-l2-rec-512
# python prune.py --batch-size 1000 --epochs 300 --dataset yahoo --num-layer 2 --shape 0 --num-neuron 512 --sparsity 0.31 --globalprune --name globalprune-yahoo-l2-rec-512

# echo "Experiment with agnews with layer-wise pruning" 1>&2
# python prune.py --batch-size 1000 --epochs 300 --dataset agnews --num-layer 2 --shape 0 --num-neuron 512 --sparsity 0.3 --globalprune --name globalprune-agnews-l2-rec-512
# python prune.py --batch-size 1000 --epochs 300 --dataset sogounews --num-layer 2 --shape 0 --num-neuron 512 --sparsity 0.3 --globalprune --name globalprune-sogounews-l2-rec-512

python prune.py --batch-size 1000 --epochs 100 --dataset dbpedia --num-layer 3 --shape 1 --num-neuron 1024 --sparsity 0.25 --globalprune --name globalprune-dbpedia-l3-con-1024
python prune.py --batch-size 1000 --epochs 100 --dataset yelp --num-layer 3 --shape 1 --num-neuron 1024 --sparsity 0.24 --globalprune --name globalprune-yelp-l3-con-1024
