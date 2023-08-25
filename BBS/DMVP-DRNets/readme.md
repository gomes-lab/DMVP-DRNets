# Files

- **config.py** :  records all constants, hyperparameters and paths.

- **train.py** : trains the model, test the model after the training and saves parameters and results

- **model.py** : provides the model structure

- **get_data.py** : provides functions to read batches from the whole dataset for training, validation or testing

- **runTrain.sh** : shell script to clean files produced before running **train.py**

- **TB.sh** : runs tensorboard

- **clean.sh** : cleans all intermediate files

- **results/** : the evaluation results of DRNets using the 12 metrics

# Abbrevations in the code:

- **nll** is the negative log-likelihood
- **prob** means probability

# Requirements

- Python 3.7
- TensorFlow 1.8
- skimage
- scikit-learn
 
# Usage 

To train and test the model:

```
bash runTrain.sh
```


To run tensorboard and monitor training progress:

```
bash TB.sh
```
