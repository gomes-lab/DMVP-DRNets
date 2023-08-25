# Usage
To run HLR-S on all datasets:

```
python subjob.py 
```

Note that the experiments were done using a computing cluster managed by slurm. You can easily modify the training command according to your own environment.


To run HLR-S on a specific dataset

```
Rscript train_NNGP.r <dataset_name> <dataset_num> 
```

# The prediction and evaluation results

The predictions will be saved in the folder **NNGP/**

To generate the 12 metric scores and save them in the folder **NNGP/results/**:

```
python eval.py
```
