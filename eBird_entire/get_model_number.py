import glob
import numpy as np
import pickle

cnt = 1

for cnt in range(1, 13):

    logs = [f for f in glob.glob("./data/small_map_data_%d/ebird_*" % (cnt), recursive=False)]

    print (len(logs))

    assert (len(logs) == 482)

    model_number = [-1 for _ in range(482)]

    for log in logs:
        print (log)
        num = int(log.strip().split("_")[4])
        print (num)
        with open(log, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                if line[0] == "the" and line[1] == "best" and line[2] == "checkpoint" and line[3] == "is":
                    mn = int(line[4])
                    model_number[num] = mn

    with open("./data/small_map_data_%d/model_number_%d.pkl" % (cnt, cnt), "wb") as f:
        pickle.dump(model_number, f)

    #print (model_number[301])
    print (model_number)



