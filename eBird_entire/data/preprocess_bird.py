import numpy as np
import os
import pickle
from sklearn import preprocessing

#file_path = "./JSDM_big_2019.06.25.csv"
#file_path = "./train_data.csv"
file_path = "./eBird_data.csv"

num_bird = 500
label_pos = 76

if __name__ == "__main__":

    with open("generated_box.pkl", "rb") as f:
        generated_box = pickle.load(f)

    box_size = 15

    sum_days = [0 for _ in range(13)]
    months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] 
    for i in range(1, 13):
        sum_days[i] = sum_days[i - 1] + months[i]
    assert (np.sum(months) == 365)
    features = [[[] for _ in range(12)] for ii in range(len(generated_box))]
    labels = [[[] for _ in range(12)] for ii in range(len(generated_box))]
    locs = [[[] for _ in range(12)] for ii in range(len(generated_box))]
    
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            print ("processing: %d/???" % (i))
            line = line.strip().split(",")
            loc = line[2:4]
            for ii in range(2):
                loc[ii] = float(loc[ii])
            lon, lat = float(loc[0]), float(loc[1])
            #print (lon, lat)
            case = []
            #if lon >= -100.0 and lon <= -50.0 and lat >= 20.0 and lat <= 55.0:
            #    case.append(1)
            #if lon >= -110.0 and lon <= -85.0 and lat >= 20.0 and lat <= 55.0:
            #    case.append(2)
            #if lon >= -137.0 and lon <= -100.0 and lat >= 20.0 and lat <= 55.0:
            #    case.append(3)
            for cur_case, (lo, la) in enumerate(generated_box):
                if lon >= lo and lon <= lo + box_size and lat >= la and lat <= la + box_size:
                    case.append(cur_case)
            if len(case) == 0:
                continue
            label = line[76:]
            feature = line[4:76]
            for ii in range(len(feature)):
                feature[ii] = float(feature[ii])
            for ii in range(len(label)):
                label[ii] = float(label[ii])
                if label[ii] >= 1 or label[ii] < 0:
                    label[ii] = 1.
                else:
                    if label[ii] != 0:
                        raise ValueError("Strange label: %f on row %d" % (label[ii], i))
            day = int(line[6])
            mon_idx = None
            for j in range(1, 13):
                if day <= sum_days[j]:
                    mon_idx = j - 1
                    break
            if day == 366:
                mon_idx = 11
            if mon_idx is None:
                print (day)
                raise ValueError("mon idx none")
            #print (day, " ", mon_idx + 1)
            for item in case:
                features[item - 1][mon_idx].append(np.array(feature))
                labels[item - 1][mon_idx].append(np.array(label))
                locs[item - 1][mon_idx].append(np.array(loc))
                    
    print ("start writting to the files")

    invalid = 0

    for i in range(len(generated_box)):
        for j in range(12):
            features[i][j] = np.array(features[i][j])
            labels[i][j] = np.array(labels[i][j])
            locs[i][j] = np.array(locs[i][j])
            if features[i][j].shape[0] <= 0:
                invalid += 1  
                continue
            features[i][j] = preprocessing.scale(features[i][j])
            print ("Location: %d, Month %d" % (i + 1, j + 1))
            print ("Features size: ", features[i][j].shape)
            print ("labels size: ", labels[i][j].shape)
            print ("location size: ", locs[i][j].shape)
            data = np.concatenate((locs[i][j], labels[i][j], features[i][j]), axis=1)
            print (data.shape)
            np.save("./small_region/small_case%d_bird_%d.npy" % (i + 1, j + 1), data)
            #np.save("/home/shared/data4esrd/new_exp/bird_%d.npy" % (i + 1, j + 1), data)

    print ("invalid: ", invalid)



