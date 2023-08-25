# eBird experiments

## Requirements

- Python 3.6
- PyTorch 1.3.1
- TensorFlow 1.8
- Torchvision 0.4.0
- Scikit-learn 0.19.2

## eBird dataset

The eBird dataset is provided in a separate package due to its size. Extract **eBird_data.csv** and **map_inference_data.csv** into the **./data/** subdirectory. 

## Experiment with data splitting

The dataset can be split into small overlapping regions to smooth the predictions. If this is desired, run the following:

```
# Preprocess the data 
cd data
python3 preprocess_folder.py
bash gen_data_split.sh
cd ..

# Training step
python3 train_split.py

# Evaluation test step
python3 test_split.py

# Prediction step on real-world data
python3 map_inference_split.py
```

## Experiment without data splitting

Alternatively, the experiments can be run without splitting the data this way:

```
# Preprocess the data 
cd data
python3 preprocess_folder.py
bash gen_data_nosplit.sh
cd ..

# Training step
python3 train_nosplit.py

# Evaluation test step
python3 test_nosplit.py

# Prediction step on real-world data
python3 map_inference_nosplit.py
```

## Log files

The training logs will be stored under **./data/train_logs/** , with filenames of the form **nosplit_ebird_x** in the nosplit case, or **ebird_y_x** in the split case, where **x** is the month and **y** is the small region number.

The nosplit evaluation test logs will be stored under **./data/test_logs/** , with filenames of the form **test_bird_x** , where **x** is the month. 

The split evaluation test logs will be stored under **./data/small_map_data[x]/**, with filenames of the form **test_bird_y_x** , where **[x]** or **x** is the month and **y** is the small region number.

The nosplit real-world test set logs will be stored under **./data/test_logs/** , with filenames of the form **test_bird_realworld_x** , where **x** is the month. 

The split real-world test set logs will be stored under **./data/small_map_data[x]/** , with filenames of the form **test_bird_realworld_y_x** , where **[x]** or **x** is the month and **y** is the small region number.

## Additional notes

- The real-world prediction step detects the best model number from the evaluation test step logs. Some regions might have a small number of data points such that no good model was found. In this case, the model number will be set to **0** or **-1**, and the total number of such regions are computed and printed as **Corrupt: xxx**. There will be corresponding error messages in the log which can be ignored, as they result from insufficient training data.

- The split test has a long runtime, so we highly recommend parallel execution.
