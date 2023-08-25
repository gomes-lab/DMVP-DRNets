import numpy as np
import os
import pickle
from sklearn import preprocessing

#ebird_file_path = "./JSDM_big_2019.06.25.csv"
#esrd_file_path = "./eSRD_3km_2017_land_only.csv"

ebird_file_path = "./eBird_data.csv"
esrd_file_path = "./map_inference_data.csv"
#esrd_file_path = "/home/fs01/df36/ttt/eSRD_3km_2017_land_only.csv"


if __name__ == "__main__":
    with open("feature_lst.pkl", "rb") as f:
        feature_lst = pickle.load(f)

    box_size = 15

    with open("generated_box.pkl", "rb") as f:
        generated_box = pickle.load(f)
    
    mon2day = {1: 15, 2: 45, 3: 74, 4: 105, 5: 135, 6: 166, 7: 196, 8: 227, 9: 258, 10: 288, 11: 319, 12: 349}

    case = []

    st = 1

    print ("box number: ", len(generated_box))
    #for mon in range(st, st + 1):
    # This model number is used to save disk space
    model_number = None
    for cnt in range(1, 13):
        #with open("./small_map_data_%d/1model_number_%d.pkl" % (cnt, cnt), "rb") as f:
        #    model_number = pickle.load(f)
        mon = cnt 
        features = [[] for _ in range(483)]
        labels = [[] for _ in range(483)]
        locs = [[] for _ in range(483)]
        with open(esrd_file_path, "r") as f:
            for i, line in enumerate(f):
                case = []
                if i == 0:
                    continue
                #print ("preprocessing: %d/???" % (i))
                line = line.strip().split(",")
                lat = float(line[83])
                lon = float(line[84])
                feature = np.zeros(72)
                label = np.zeros(500)
                loc = [lat, lon]
                #if lon >= -100.0 and lon <= -50.0 and lat >= 20.0 and lat <= 55.0:
                #    case.append(1)
                #if lon >= -110.0 and lon <= -85.0 and lat >= 20.0 and lat <= 55.0:
                #    case.append(2)
                #if lon >= -137.0 and lon <= -100.0 and lat >= 20.0 and lat <= 55.0:
                #    case.append(3)
                for cur_case, (lo, la) in enumerate(generated_box):
                    if lon >= lo and lon <= lo + box_size and lat >= la and lat <= la + box_size:
                        #if model_number[cur_case - 1] > 1: 
                        case.append(cur_case)
                if len(case) == 0:
                    continue
                #if lat < 30.0 or lat > 50. or lon < -85.0 or lon > -65.0:
                #    continue
                for j, idx in enumerate(feature_lst):
                    feature[j + 8] = float(line[idx])
                feature[0] = 0
                feature[1] = 2008.0
                feature[2] = mon2day[mon]
                feature[3] = 7.0
                feature[4] = 1.0
                feature[5] = 1.0
                feature[6] = 1.0
                feature[7] = 2.0
                for item in case:
                    features[item - 1].append(feature)
                    labels[item - 1].append(label)
                    locs[item - 1].append(np.array(loc))
        features = np.array(features)
        labels = np.array(labels)
        locs = np.array(locs)
        for item in range(len(features)):
            if len(features[item - 1]) <= 0:
                continue
            else:
                print ("has something")
            features[item - 1] = np.array(features[item - 1])
            labels[item - 1] = np.array(labels[item - 1])
            locs[item - 1] = np.array(locs[item - 1])
            features[item - 1] = preprocessing.scale(features[item - 1])
            print ("idx: %d" % (item))
            print ("Month %d" % (mon))
            print ("Features size: ", features[item - 1].shape)
            print ("labels size: ", labels[item - 1].shape)
            print ("location size: ", locs[item - 1].shape)
            data = np.concatenate((locs[item - 1], labels[item - 1], features[item - 1]), axis=1)
            print ("data size: ", data.shape)
            np.save("./small_esrd/small_daniel_esrd_%d_%d.npy" % (item, mon), data)





























