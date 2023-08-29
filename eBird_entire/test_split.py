import os
import pickle

os.system("python3 get_model_number.py")

for cnt in range(1, 13):
    with open("./data/small_map_data_%d/model_number_%d.pkl" % (cnt, cnt), "rb") as f:
        model_num = pickle.load(f)

    assert (len(model_num) == 482)

    cur = -1
    corrupt = 0

    for i in range(482):
        if model_num[i] == -1 or model_num[i] == 0:
            corrupt += 1
            cur += 1
            continue
        for j in range(1):
            j = cnt 
            cur += 1
            print (cur)
            j_name = "test_bird_%d_%d" % (i, j)
            command = "python3 test.py --checkpoint_path ./data/small_model_mon%d/model_%d_%d/model-%d --summary_dir ./summary_%d_%d --visual_dir ./vis_%d_%d --data_dir ./data/small_region/small_case%d_bird_%d.npy --test_idx ./data/small_region/small_case%d_bird_test_idx_%d.npy --r_dim 500 --r_max_dim 500 --mon %d --case %d | tee ./data/small_map_data%d/%s" % (cnt, i, j, model_num[cur], i, j, i, j, i, j, i, j, j, i, cnt, j_name)

            print (command)
            os.system(command)

    print ("corrupt: ", corrupt)
