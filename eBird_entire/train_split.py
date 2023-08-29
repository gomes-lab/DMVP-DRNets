#python3 train.py --model_dir ./tmpmodel --summary_dir ./tmpsum --visual_dir ./tmpvis --data_dir ../data/new_bird_1.npy --train_idx ../data/new_bird_train_idx_1.npy --valid_idx ../data/new_bird_val_idx_1.npy --test_idx ../data/new_bird_test_idx_1.npy --r_dim 500 --r_max_dim 500

import os

for i in range(1, 483):
    for j in range(1, 13):
        command = "python3 train.py --model_dir ./data/small_model_mon%d/model_%d_%d --summary_dir ./summary_%d_%d --visual_dir ./vis_%d_%d --data_dir ./data/small_region/small_case%d_bird_%d.npy --train_idx ./data/small_region/small_case%d_bird_train_idx_%d.npy --valid_idx ./data/small_region/small_case%d_bird_val_idx_%d.npy --test_idx ./data/small_region/small_case%d_bird_test_idx_%d.npy --r_dim 500 --r_max_dim 500 | tee ./data/small_map_data_%d/ebird_%d_%d" % (j, i, j, i, j, i, j, i, j, i, j, i, j, i, j, j, i, j)
        print (command)
        os.system(command)

