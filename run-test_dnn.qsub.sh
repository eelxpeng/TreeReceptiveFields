#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

# python test_dnn.py --batch-size 256 --epochs 50 --lr 0.001 --name agnews-dnn-larger
python test_dnn.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --name dbpedia-dnn
python test_dnn.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --name yelp-dnn
python test_dnn.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --name yahoo-dnn

