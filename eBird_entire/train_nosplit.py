#python3 train.py --model_dir ./tmpmodel --summary_dir ./tmpsum --visual_dir ./tmpvis --data_dir ../data/daniel_new_bird_1.npy --train_idx ../data/daniel_new_bird_train_idx_1.npy --valid_idx ../data/daniel_new_bird_val_idx_1.npy --test_idx ../data/daniel_new_bird_test_idx_1.npy --r_dim 500 --r_max_dim 500

import os

for i in range(1):
    for j in range(1, 13):
        job_name = "ebird_%d" % (j)
        command = "python3 train.py --model_dir ./data/new_model%d/ --summary_dir ./new_summary_%d --visual_dir ./new_vis_%d --data_dir ./data/small_region/bird_%d.npy --train_idx ./data/small_region/bird_train_idx_%d.npy --valid_idx ./data/small_region/bird_val_idx_%d.npy --test_idx ./data/small_region/bird_test_idx_%d.npy --r_dim 500 --r_max_dim 500 | tee ./data/train_logs/nosplit_ebird_%d" % (j, j, j, j, j, j, j, j)
        print (command)
        os.system(command)

