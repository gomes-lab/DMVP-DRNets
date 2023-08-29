import numpy as np
import os
import pickle

train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

with open("generated_box.pkl", "rb") as f:
    generated_box = pickle.load(f)

for ii in range(len(generated_box)):
    for jj in range(1, 13):
        if not os.path.exists("./small_region/small_case%d_bird_%d.npy" % (ii, jj)):
            continue
        a = np.load("./small_region/small_case%d_bird_%d.npy" % (ii, jj))
        num = a.shape[0]

        train_idx = []
        val_idx = []
        test_idx = []

        idx = np.arange(num)
        np.random.shuffle(idx)

        idx1 = int(num * train_frac)
        idx2 = idx1 + int(num * val_frac)

        for i in range(idx1):
            train_idx.append(idx[i])
        for i in range(idx1, idx2):
            val_idx.append(idx[i])
        for i in range(idx2, num):
            test_idx.append(idx[i])
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
        print ("train: ", train_idx.shape)
        print ("val: ", val_idx.shape)
        print ("test: ", test_idx.shape)

        np.save("./small_region/small_case%d_bird_train_idx_%d.npy" % (ii, jj), train_idx)
        np.save("./small_region/small_case%d_bird_val_idx_%d.npy" % (ii, jj), val_idx)
        np.save("./small_region/small_case%d_bird_test_idx_%d.npy" % (ii, jj), test_idx)

