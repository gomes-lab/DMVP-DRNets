import glob
import numpy as np
import pickle

cnt = 1

logs = [f for f in glob.glob("./data/train_logs/nosplit_ebird_*", recursive=False)]
assert (len(logs) == 12)

model_number = [-1 for _ in range(13)]

for log in logs:
    print (log)
    num = int(log.strip().split("_")[3])
    print (num)
    with open(log, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            if line[0] == "the" and line[1] == "best" and line[2] == "checkpoint" and line[3] == "is":
                mn = int(line[4])
                model_number[num] = mn

with open("./data/small_region/model_number.pkl", "wb") as f:
    pickle.dump(model_number, f)

print (model_number)



