import numpy as np
import os
import pickle
from sklearn import preprocessing

#ebird_file_path = "./data/JSDM_big_2019.06.25.csv"
#esrd_file_path = "./data/eSRD_3km_2017_land_only.csv"
#ebird_file_path = "./train_data.csv"
#esrd_file_path = "./test_data.csv"
ebird_file_path = "./eBird_data.csv"
esrd_file_path = "./map_inference_data.csv"


if __name__ == "__main__":
    with open("feature_lst.pkl", "rb") as f:
        feature_lst = pickle.load(f)

    box_size = 15

    with open("generated_box.pkl", "rb") as f:
        generated_box = pickle.load(f)
    
    mon2day = {1: 15, 2: 45, 3: 74, 4: 105, 5: 135, 6: 166, 7: 196, 8: 227, 9: 258, 10: 288, 11: 319, 12: 349}

    case = []

    st = 1

    #for mon in range(st, st + 1):
    model_number = None
    for cnt in range(1, 13):
        mon = cnt 
        features = []
        labels = []
        locs = []
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
                features.append(feature)
                labels.append(label)
                locs.append(loc)
        features = np.array(features)
        labels = np.array(labels)
        locs = np.array(locs)

        #for item in range(len(features)):
        for _ in range(1):
            features = preprocessing.scale(features)
            print ("Month %d" % (mon))
            print ("Features size: ", features.shape)
            print ("labels size: ", labels.shape)
            print ("location size: ", locs.shape)
            data = np.concatenate((locs, labels, features), axis=1)
            print ("data size: ", data.shape)
            #np.save("/home/shared/data4esrd/small_esrd/small_esrd_%d_%d.npy" % (item, mon), data)
            np.save("./small_esrd/esrd_%d.npy" % (mon), data)





























