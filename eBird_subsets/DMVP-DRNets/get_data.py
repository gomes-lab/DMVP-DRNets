import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import config 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import pandas as pd
FLAGS = tf.app.flags.FLAGS


class Data_loader():
    def __init__(self, exp, num, use_S = False):
        train_X = self.extract_data("../DATA/Xt_%s_%s.csv" % (num, exp))
        test_X = self.extract_data("../DATA/Xv_%s_%s.csv" % (num, exp))

        train_Y = self.extract_data("../DATA/Yt_%s_%s.csv" % (num, exp))
        test_Y = self.extract_data("../DATA/Yv_%s_%s.csv" % (num, exp))

        train_S = self.extract_data("../DATA/St_%s_%s.csv" % (num, exp))
        test_S = self.extract_data("../DATA/Sv_%s_%s.csv" % (num, exp))

        if (use_S):
            train_X = np.concatenate([train_X, train_S], axis = 1)
            test_X = np.concatenate([test_X, test_S], axis = 1)

        self.X_mean = np.mean(train_X, axis = 0) 
        self.X_std = np.maximum(np.std(train_X, axis = 0), 1e-9)

        self.X = np.concatenate([train_X, test_X], axis = 0)
        self.Y = np.concatenate([train_Y, test_Y], axis = 0)
        self.S = np.concatenate([train_S, test_S], axis = 0)

        print("occurance_rate", np.mean(np.sum(self.Y, axis = 1)))

        self.train_idx = np.arange(len(train_X))
        self.test_idx = np.arange(len(test_X)) + len(train_X)


    def get_indices(self):
        return self.train_idx, self.test_idx, self.X.shape[1], self.Y.shape[1]
        
        
    def extract_data(self, path):
        df = pd.read_csv(path, header = None)
        x = df.values
        print("load %s"%path, x.shape)
        return x
   
    def get_Y(self, my_order):
        output = []
        for i in my_order:
            x = np.copy(self.Y[i])
            output.append(x)

        output = np.array(output, dtype="int") 
        return output

    def get_X(self, my_order, normalize = True):
        output = []
            
        for i in my_order:
            x = np.copy(self.X[i])
            if (normalize):
                x = (x - self.X_mean) / self.X_std
            output.append(x)

        output = np.array(output, dtype="float32") 
        return output

    def get_loc(self, my_order):

        output = []
        for i in my_order:
            output.append(np.copy(self.S[i]))
        output = np.array(output, dtype="float32") 
        return output



class Log():
    def __init__(self):
        self.Vars = {}

    def add(self, name, value, weight = 1):
        if (not name in self.Vars):
            self.Vars[name] = []
        self.Vars[name].append([value * weight, weight])

    def get_var_names(self):
        return sorted(self.Vars.keys())

    def get_means(self, names, clean = True):
        res = []
        if (not isinstance(names, list)):
            print("<names> should be a list")
            return 

        for name in names:
            data = self.Vars[name]

            if (len(data) == 0):
                avg = float('nan')
            else:
                data = np.asarray(data, dtype = "float32")
                avg = np.sum(data[:, 0]) / np.maximum(np.sum(data[:, 1]), 1e-9)

            res.append([name, avg])

            if (clean):
                self.Vars[name] = []

        return res

    def get_mean(self, name, clean = False):
        data = self.Vars[name]
        if (len(data) == 0):
            avg = float('nan')
        else:
            data = np.asarray(data, dtype = "float32")
            avg = np.sum(data[:, 0]) / np.maximum(np.sum(data[:, 1]), 1e-9)
            
        if (clean):
            self.Vars[name] = []

        return avg



######################################################################
