#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e
# #$ -l h=client110

# Agnews
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 3 --name agnews-treeconvfc-layer-3 > logs/log-agnews-treeconvfc-layer-3-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 1 --name agnews-treeconvfc-layer-1 > logs/log-agnews-treeconvfc-layer-1-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 2 --name agnews-treeconvfc-layer-2 > logs/log-agnews-treeconvfc-layer-2-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 4 --name agnews-treeconvfc-layer-4 > logs/log-agnews-treeconvfc-layer-4-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 5 --name agnews-treeconvfc-layer-5 > logs/log-agnews-treeconvfc-layer-5-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 6 --name agnews-treeconvfc-layer-6 > logs/log-agnews-treeconvfc-layer-6-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 7 --name agnews-treeconvfc-layer-7 > logs/log-agnews-treeconvfc-layer-7-run-2

# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 3 --name agnews-treeconvfc-layer-3 > logs/log-agnews-treeconvfc-layer-3-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 1 --name agnews-treeconvfc-layer-1 > logs/log-agnews-treeconvfc-layer-1-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 2 --name agnews-treeconvfc-layer-2 > logs/log-agnews-treeconvfc-layer-2-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 4 --name agnews-treeconvfc-layer-4 > logs/log-agnews-treeconvfc-layer-4-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 5 --name agnews-treeconvfc-layer-5 > logs/log-agnews-treeconvfc-layer-5-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 6 --name agnews-treeconvfc-layer-6 > logs/log-agnews-treeconvfc-layer-6-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 7 --name agnews-treeconvfc-layer-7 > logs/log-agnews-treeconvfc-layer-7-run-3

# DBPedia
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 3 --name dbpedia-treeconvfc-layer-3 > logs/log-dbpedia-treeconvfc-layer-3-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 1 --name dbpedia-treeconvfc-layer-1 > logs/log-dbpedia-treeconvfc-layer-1-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 2 --name dbpedia-treeconvfc-layer-2 > logs/log-dbpedia-treeconvfc-layer-2-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 4 --name dbpedia-treeconvfc-layer-4 > logs/log-dbpedia-treeconvfc-layer-4-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 5 --name dbpedia-treeconvfc-layer-5 > logs/log-dbpedia-treeconvfc-layer-5-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 6 --name dbpedia-treeconvfc-layer-6 > logs/log-dbpedia-treeconvfc-layer-6-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 7 --name dbpedia-treeconvfc-layer-7 > logs/log-dbpedia-treeconvfc-layer-7-run-2

# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 3 --name dbpedia-treeconvfc-layer-3 > logs/log-dbpedia-treeconvfc-layer-3-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 1 --name dbpedia-treeconvfc-layer-1 > logs/log-dbpedia-treeconvfc-layer-1-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 2 --name dbpedia-treeconvfc-layer-2 > logs/log-dbpedia-treeconvfc-layer-2-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 4 --name dbpedia-treeconvfc-layer-4 > logs/log-dbpedia-treeconvfc-layer-4-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 5 --name dbpedia-treeconvfc-layer-5 > logs/log-dbpedia-treeconvfc-layer-5-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 6 --name dbpedia-treeconvfc-layer-6 > logs/log-dbpedia-treeconvfc-layer-6-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 7 --name dbpedia-treeconvfc-layer-7 > logs/log-dbpedia-treeconvfc-layer-7-run-3

# Yelp
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 3 --name yelp-treeconvfc-layer-3 > logs/log-yelp-treeconvfc-layer-3-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 1 --name yelp-treeconvfc-layer-1 > logs/log-yelp-treeconvfc-layer-1-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 2 --name yelp-treeconvfc-layer-2 > logs/log-yelp-treeconvfc-layer-2-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 4 --name yelp-treeconvfc-layer-4 > logs/log-yelp-treeconvfc-layer-4-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 5 --name yelp-treeconvfc-layer-5 > logs/log-yelp-treeconvfc-layer-5-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 6 --name yelp-treeconvfc-layer-6 > logs/log-yelp-treeconvfc-layer-6-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 7 --name yelp-treeconvfc-layer-7 > logs/log-yelp-treeconvfc-layer-7-run-1

# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 3 --name yelp-treeconvfc-layer-3 > logs/log-yelp-treeconvfc-layer-3-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 1 --name yelp-treeconvfc-layer-1 > logs/log-yelp-treeconvfc-layer-1-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 2 --name yelp-treeconvfc-layer-2 > logs/log-yelp-treeconvfc-layer-2-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 4 --name yelp-treeconvfc-layer-4 > logs/log-yelp-treeconvfc-layer-4-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 5 --name yelp-treeconvfc-layer-5 > logs/log-yelp-treeconvfc-layer-5-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 6 --name yelp-treeconvfc-layer-6 > logs/log-yelp-treeconvfc-layer-6-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 7 --name yelp-treeconvfc-layer-7 > logs/log-yelp-treeconvfc-layer-7-run-2

# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 3 --name yelp-treeconvfc-layer-3 > logs/log-yelp-treeconvfc-layer-3-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 1 --name yelp-treeconvfc-layer-1 > logs/log-yelp-treeconvfc-layer-1-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 2 --name yelp-treeconvfc-layer-2 > logs/log-yelp-treeconvfc-layer-2-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 4 --name yelp-treeconvfc-layer-4 > logs/log-yelp-treeconvfc-layer-4-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 5 --name yelp-treeconvfc-layer-5 > logs/log-yelp-treeconvfc-layer-5-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 6 --name yelp-treeconvfc-layer-6 > logs/log-yelp-treeconvfc-layer-6-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 7 --name yelp-treeconvfc-layer-7 > logs/log-yelp-treeconvfc-layer-7-run-3

# Sogounews
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 3 --name sogounews-treeconvfc-layer-3 > logs/log-sogounews-treeconvfc-layer-3-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 1 --name sogounews-treeconvfc-layer-1 > logs/log-sogounews-treeconvfc-layer-1-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 2 --name sogounews-treeconvfc-layer-2 > logs/log-sogounews-treeconvfc-layer-2-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 4 --name sogounews-treeconvfc-layer-4 > logs/log-sogounews-treeconvfc-layer-4-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 5 --name sogounews-treeconvfc-layer-5 > logs/log-sogounews-treeconvfc-layer-5-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 6 --name sogounews-treeconvfc-layer-6 > logs/log-sogounews-treeconvfc-layer-6-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 7 --name sogounews-treeconvfc-layer-7 > logs/log-sogounews-treeconvfc-layer-7-run-1

# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 3 --name sogounews-treeconvfc-layer-3 > logs/log-sogounews-treeconvfc-layer-3-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 1 --name sogounews-treeconvfc-layer-1 > logs/log-sogounews-treeconvfc-layer-1-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 2 --name sogounews-treeconvfc-layer-2 > logs/log-sogounews-treeconvfc-layer-2-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 4 --name sogounews-treeconvfc-layer-4 > logs/log-sogounews-treeconvfc-layer-4-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 5 --name sogounews-treeconvfc-layer-5 > logs/log-sogounews-treeconvfc-layer-5-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 6 --name sogounews-treeconvfc-layer-6 > logs/log-sogounews-treeconvfc-layer-6-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 7 --name sogounews-treeconvfc-layer-7 > logs/log-sogounews-treeconvfc-layer-7-run-2

# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 3 --name sogounews-treeconvfc-layer-3 > logs/log-sogounews-treeconvfc-layer-3-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 1 --name sogounews-treeconvfc-layer-1 > logs/log-sogounews-treeconvfc-layer-1-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 2 --name sogounews-treeconvfc-layer-2 > logs/log-sogounews-treeconvfc-layer-2-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 4 --name sogounews-treeconvfc-layer-4 > logs/log-sogounews-treeconvfc-layer-4-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 5 --name sogounews-treeconvfc-layer-5 > logs/log-sogounews-treeconvfc-layer-5-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 6 --name sogounews-treeconvfc-layer-6 > logs/log-sogounews-treeconvfc-layer-6-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 7 --name sogounews-treeconvfc-layer-7 > logs/log-sogounews-treeconvfc-layer-7-run-3

# Yahoo
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 3 --name yahoo-treeconvfc-layer-3 > logs/log-yahoo-treeconvfc-layer-3-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 1 --name yahoo-treeconvfc-layer-1 > logs/log-yahoo-treeconvfc-layer-1-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 2 --name yahoo-treeconvfc-layer-2 > logs/log-yahoo-treeconvfc-layer-2-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 4 --name yahoo-treeconvfc-layer-4 > logs/log-yahoo-treeconvfc-layer-4-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 5 --name yahoo-treeconvfc-layer-5 > logs/log-yahoo-treeconvfc-layer-5-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 6 --name yahoo-treeconvfc-layer-6 > logs/log-yahoo-treeconvfc-layer-6-run-1
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 7 --name yahoo-treeconvfc-layer-7 > logs/log-yahoo-treeconvfc-layer-7-run-1

# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 3 --name yahoo-treeconvfc-layer-3 > logs/log-yahoo-treeconvfc-layer-3-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 1 --name yahoo-treeconvfc-layer-1 > logs/log-yahoo-treeconvfc-layer-1-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 2 --name yahoo-treeconvfc-layer-2 > logs/log-yahoo-treeconvfc-layer-2-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 4 --name yahoo-treeconvfc-layer-4 > logs/log-yahoo-treeconvfc-layer-4-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 5 --name yahoo-treeconvfc-layer-5 > logs/log-yahoo-treeconvfc-layer-5-run-2
python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 6 --name yahoo-treeconvfc-layer-6 > logs/log-yahoo-treeconvfc-layer-6-run-2
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 7 --name yahoo-treeconvfc-layer-7 > logs/log-yahoo-treeconvfc-layer-7-run-2

# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 3 --name yahoo-treeconvfc-layer-3 > logs/log-yahoo-treeconvfc-layer-3-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 1 --name yahoo-treeconvfc-layer-1 > logs/log-yahoo-treeconvfc-layer-1-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 2 --name yahoo-treeconvfc-layer-2 > logs/log-yahoo-treeconvfc-layer-2-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 4 --name yahoo-treeconvfc-layer-4 > logs/log-yahoo-treeconvfc-layer-4-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 5 --name yahoo-treeconvfc-layer-5 > logs/log-yahoo-treeconvfc-layer-5-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 6 --name yahoo-treeconvfc-layer-6 > logs/log-yahoo-treeconvfc-layer-6-run-3
# python test_treeconvfc-bin-layer.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --num-layer 7 --name yahoo-treeconvfc-layer-7 > logs/log-yahoo-treeconvfc-layer-7-run-3




# python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --name dbpedia-treeconvfc
# python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --name yelp-treeconvfc
# python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --name yahoo-treeconvfc-3
# python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --name sogounews-treeconvfc
