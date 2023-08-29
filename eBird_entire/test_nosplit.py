import os
import pickle

os.system("python3 nosplit_get_model_number.py")

with open("./data/small_region/model_number.pkl", "rb") as f:
    model_num = pickle.load(f)

for cnt in range(1, 13):
    assert (len(model_num) == 13)

    cur = -1
    corrupt = 0

    for i in range(1):
        if model_num[cnt] == -1 or model_num[cnt] == 0:
            corrupt += 1
            cur += 1
            continue
        for j in range(1):
            j = cnt 
            cur += 1
            print (cur)
            j_name = "test_bird_%d" % (j)
            command = "python3 test.py --checkpoint_path ./data/new_model%d/model-%d --summary_dir ./new_summary_%d --visual_dir ./new_vis_%d --data_dir ./data/small_region/bird_%d.npy --test_idx ./data/small_region/bird_test_idx_%d.npy --r_dim 500 --r_max_dim 500 --mon %d | tee ./data/test_logs/%s" % (cnt, model_num[cnt], j, j, j, j, cnt, j_name)

            print (command)
            os.system(command)

    print ("corrupt: ", corrupt)
