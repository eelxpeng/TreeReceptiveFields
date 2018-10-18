# Tree Receptive Fields

This repository is associated with the following paper:

Xiaopeng Li*, Zhourong Chen* and Nevin L. Zhang. Building Sparse Deep Feedforward Networks using Tree Receptive Fields. IJCAI 2018.

## Environment Requirements

*python3.6
*pytorch>=0.4.0
*numpy
*scipy
*sklearn

## Running

The dataset for Agnews is put under `dataset/' for demo. To run the program with Agnews dataset, simply run

```console
bash run-test_treeconvfc-agnews.qsub.sh
```

The driving script for TRFNet with global neurons is `test_treeconvfc.py`. The driving script for TRFNet with TRF neurons only is `test_treeconv.py`. 

