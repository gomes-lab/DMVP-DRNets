import numpy as np
import pandas as pd
from sklearn.decomposition import PCA 
import sys, os
def save(name, exp_name, exp_num, x):
    full_name = name+"_%s_%s.csv"%(exp_num, exp_name)
    print(full_name, x.shape)
    np.savetxt(full_name, x, delimiter = ",")

def PCA_transform(X, n_f):
    pca = PCA(n_components = n_f, whiten = True)
    newX = pca.fit_transform(X)
    return newX

def get_sampled_species(n_sp):
    df = pd.read_csv("Grouped_1.csv", index_col = 0)
    #print(df.shape)
    D = {}
    for i in range(df.shape[0]):
        gp = df["HIGHLIGHT_GROUP"].iloc[i]
        prev = df["prev"].iloc[i]
        name = df["SCI_NAME_KEY"].iloc[i]
        if (not gp in D):
            D[gp] = []
        D[gp].append([prev, name])

    for k in D.keys():
        D[k].sort(key = lambda x: -x[0])
    
    keys = list(D.keys())
    keys.sort(key = lambda x: len(D[x]))
    
    res_sp = n_sp
    sp_lst = []
    for i in range(len(keys)):
        key = keys[i]
        n_gp = len(D[key])
        n_ext = min(n_gp, (res_sp - 1) //(len(keys) - i) + 1)
        for j in range(n_ext):
            sp_lst.append(D[key][j][1])
        res_sp -= n_ext
    print("get n_sp:", len(sp_lst))
    return sp_lst

def generate_data(df, n_species, n_feature, n_train, n_test, exp_name, exp_num):

    sp_lst = get_sampled_species(n_species)

    columns = list(df.columns)
    f_lst = []
    for col in columns[4:]:
        if (col == "Zenaida_macroura"):
            break
        f_lst.append(col)

    #print("n_feature", len(f_lst))


    S = df[["LONGITUDE", "LATITUDE"]].values + np.random.randn(df.shape[0], 2) * 1e-4
    X = df[f_lst].values
    X = PCA_transform(X, n_feature)
    Y = df[sp_lst].values
    Y = (Y != 0).astype("int")

    print(S.shape, X.shape, Y.shape)

    idx = [] 
    day = df["DAY"].values
    for i in range(df.shape[0]):
        if ( 30 * 4 < day[i] <= 30 * 6):
            idx.append(i)
    print("idx", len(idx))
    np.random.shuffle(idx)

    train_idx = idx[n_test:n_test + n_train]
    test_idx = idx[:n_test]

    St = S[train_idx]
    Sv = S[test_idx]
    Xt = X[train_idx]
    Xv = X[test_idx]
    Yt = Y[train_idx]
    Yv = Y[test_idx]
    save("St", exp_name, exp_num, St)
    save("Sv", exp_name, exp_num, Sv)
    save("Xt", exp_name, exp_num, Xt)
    save("Xv", exp_name, exp_num, Xv)
    save("Yt", exp_name, exp_num, Yt)
    save("Yv", exp_name, exp_num, Yv)


#df = pd.read_csv("/mnt/beegfs/bulk/mirror/dichen/JSDM_big_sample.csv")
#df = pd.read_csv("/mnt/beegfs/bulk/mirror/dichen/JSDM_big_2019.06.25.csv")
df = pd.read_csv("eBird_data.csv")
print(df.shape)

n_dataset = 5
n_species = [200] * n_dataset #int(sys.argv[1]) #100
n_feature = [5, 20, 72, 72, 5] #int(sys.argv[2]) #5
n_test = [1000] * 5 #int(sys.argv[3]) #1000
n_train = [1000, 10000, 100000, 1000000, 10000] #int(sys.argv[4]) #1000
exp_name = ["ebird"] * n_dataset
exp_num = list(range(1, n_dataset + 1)) #int(sys.argv[5]) #1

np.random.seed(19950420)

for i in range(n_dataset):
    generate_data(df, n_species[i], n_feature[i], n_train[i], n_test[i], exp_name[i], exp_num[i])
