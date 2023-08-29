import numpy as np
import os
import pickle

with open("generated_box.pkl", "rb") as f:
    generated_box = pickle.load(f)

pt2cnt = {}

for j in range(1, 13):
    pt2cnt = {}
    with open("box_info_%d.csv" % (j), "w") as f:
        for i in range(482):
            if not os.path.exists("/home/shared/data4esrd/small_region/small_case%d_bird_%d.npy" % (i + 1, j)):
                #f.write("empty" + "\n")
                continue
            a = np.load("/home/shared/data4esrd/small_region/small_case%d_bird_%d.npy" % (i + 1, j))
            locs = set()
            #f.write(str(i) + "," + str(generated_box[i][0]) + "," + str(generated_box[i][1]) + "," + str(a.shape[0]) + "\n") 
            for k in range(a.shape[0]):
                loc = (a[k][0], a[k][1])
                if loc in locs:
                    continue
                else:
                    locs.add(loc)
                if loc in pt2cnt:
                    pt2cnt[loc] += 1
                else:
                    pt2cnt[loc] = 1
    max_loc = None
    max_cnt = 0
    for k, v in pt2cnt.items():
        if v > max_cnt:
            max_cnt = v
            max_loc = k
    print ("max points of month: %d is %d" % (j, max_cnt))
    print ("loc: ", max_loc)


