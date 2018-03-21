#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

# python test_treeconvfc.py --batch-size 256 --epochs 50 --lr 0.001 --name agnews-treeconvfc
# python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --name dbpedia-treeconvfc
# python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --name yelp-treeconvfc
python test_finetune.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.01 --file checkpoint/ckpt-yahoo-treeconvfc-structure.t7 --name yahoo-finetune-step-0.01
# python test_finetune.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --name sogounews-treeconvfc
