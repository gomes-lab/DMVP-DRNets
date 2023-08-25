import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import model2 as model
import get_data 
import config 
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import math
import urllib
from pyheatmap.heatmap import HeatMap
import seaborn as sns

FLAGS = tf.app.flags.FLAGS

def analysis(species, indiv_prob, input_label, printNow = False):
    

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    pred_label = np.greater(indiv_prob, FLAGS.threshold).astype(int)

    #print input_label.shape 
    for j in range(input_label.shape[1]):
        for i in range(input_label.shape[0]):
            if (pred_label[i][j]==1 and input_label[i][j]==1):
                TP+=1
            if (pred_label[i][j]==1 and input_label[i][j]==0):
                FP+=1
            if (pred_label[i][j]==0 and input_label[i][j]==0):
                TN+=1
            if (pred_label[i][j]==0 and input_label[i][j]==1):
                FN+=1

    N = (TP+TN+FN+FP)*1.0
    eps = 1e-6
    precision = 1.0 * TP / (TP + FP + eps)
    recall = 1.0 * TP / (TP + FN + eps) 
    Accuracy =  1.0*(TN+TP)/(N)
    F1 = 2.0 * precision * recall / (precision + recall + eps)
    #print "F2:", (1+4)*precision*recall/(4*precision+recall)

    occurrence = np.mean(input_label)
    auc = roc_auc_score(input_label, indiv_prob)

    indiv_prob = np.reshape(indiv_prob, (-1))
    input_label = np.reshape(input_label, (-1))

    new_auc = roc_auc_score(input_label, indiv_prob)

    ap = average_precision_score(input_label,indiv_prob)

    if (printNow):
        print ("\nThis is the analysis of #%s species:"%species)
        print ("occurrence rate:", occurrence)
        print ("Overall \tauc=%.6f\tnew_auc=%.6f\tap=%.6f" % (auc, new_auc, ap))    
        print ("F1:", F1 )
        print ("Accuracy:", Accuracy)
        print ("Precision:", precision)
        print ("Recall:", recall)
        print ("TP=%f, TN=%f, FN=%f, FP=%f"%(TP/N, TN/N, FN/N, FP/N))
        print (" ")
    return occurrence, auc, F1, Accuracy, new_auc, ap,  precision, recall, TP/N, TN/N, FN/N, FP/N,

def main(_):

    print ('reading npy...')

    data = np.load(FLAGS.data_dir)
    if "esrd" in FLAGS.data_dir:
        test_idx = [i for i in range(data.shape[0])]
    else:
        test_idx = np.load(FLAGS.test_idx)

    print ('reading completed')

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)

    print ('building network...')

    classifier = model.MODEL(is_training=False)
    global_step = tf.Variable(0,name='global_step',trainable=False)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess,FLAGS.checkpoint_path)

    print ('restoring from '+FLAGS.checkpoint_path)

    with tf.variable_scope("r_mu", reuse=True):
        last_layer = tf.get_variable("weights")

    feature_embedding = sess.run([last_layer])[0]
    feature_embedding = np.transpose(feature_embedding)
    #print (feature_embedding.shape)
    #np.save("./small_new_bird_feature_embedding.npy", feature_embedding)
    os.makedirs(FLAGS.visual_dir, exist_ok=True)
    np.save(os.path.join(FLAGS.visual_dir, "feature_emb_%d" % (FLAGS.mon)), feature_embedding)

    def test_step():
        print ('Testing...')
        all_nll_loss = 0
        all_l2_loss = 0
        all_total_loss = 0
        all_marginal_loss = 0

        all_indiv_prob = []
        all_label = []

        sigma=[]
        real_batch_size=min(FLAGS.testing_size, len(test_idx))
        avg_cor=0
        cnt_cor=0
        
        N_test_batch = int( (len(test_idx)-1)/real_batch_size )+1
        #print N_test_batch
        prob_res = []
        prob_res_sample = []
        loc_res = []

        for i in range(N_test_batch):

            print ("%.1f%% completed" % (i*100.0/N_test_batch))

            start = real_batch_size*i
            end = min(real_batch_size*(i+1), len(test_idx))

            #input_image= get_data.get_image(images, test_idx[start:end])
            input_loc  = get_data.get_loc(data, test_idx[start:end])
            input_nlcd = get_data.get_nlcd(data,test_idx[start:end])
            input_label = get_data.get_label(data,test_idx[start:end])
            #print (input_loc.shape)
            #print (input_loc[2])
            #exit(0)

            feed_dict={}
            feed_dict[classifier.input_nlcd]=input_nlcd
            feed_dict[classifier.input_label]=input_label
            #feed_dict[classifier.input_image]=input_image
            feed_dict[classifier.keep_prob]=1.0
            

            nll_loss, marginal_loss, l2_loss, total_loss, indiv_prob, indiv_prob_sample, covariance = sess.run([classifier.nll_loss, classifier.marginal_loss, classifier.l2_loss, \
                classifier.total_loss, classifier.indiv_prob, classifier.indiv_prob2, classifier.covariance],feed_dict)
            #print (input_loc.shape)
            #print (indiv_prob.shape)
            ttt = []
            for ii in range(indiv_prob.shape[0]):
                prob_res.append(indiv_prob[ii])
                prob_res_sample.append(indiv_prob_sample[ii])
                loc_res.append(input_loc[ii])
            for (item1, item2) in zip(prob_res[len(prob_res) - 1], prob_res_sample[len(prob_res_sample) - 1]):
                ttt.append((float(item1) - float(item2)) ** 2)
            print (np.mean(ttt))
            
            all_nll_loss += nll_loss*(end-start)
            all_marginal_loss += marginal_loss * (end - start)
            all_l2_loss += l2_loss*(end-start)
            all_total_loss += total_loss*(end-start)

            if (all_indiv_prob == []):
                all_indiv_prob = indiv_prob
            else:
                all_indiv_prob = np.concatenate((all_indiv_prob, indiv_prob))

            if (all_label == []):
                all_label = input_label
            else:
                all_label = np.concatenate((all_label, input_label))


        
        #print "Overall occurrence ratio: %f"%(np.mean(all_label))
        
        nll_loss = all_nll_loss / len(test_idx)
        l2_loss = all_l2_loss / len(test_idx)
        marginal_loss = all_marginal_loss / len(test_idx)
        total_loss = all_total_loss / len(test_idx)

        time_str = datetime.datetime.now().isoformat()

        print ("performance on test_set: nll_loss=%.6f\tmarginal_loss:%.6f\tl2_loss=%.6f\ttotal_loss=%.6f \n%s" % (nll_loss, marginal_loss, l2_loss, total_loss, time_str)  )
        
        #print FLAGS.visual_dir+"cov"
        np.save(os.path.join(FLAGS.visual_dir, "cov"), covariance)
        return all_indiv_prob, all_label, prob_res, loc_res

        
    
    #file_path = "./data/JSDM_big_2019.06.25.csv"
    #file_path = "./data/train_data.csv"
    file_path = "./data/eBird_data.csv"
    spe_name = None
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip().split(",")
            if i == 0:
                spe_name = line[76:]
            else:
                break
    assert (len(spe_name) == 500)
    print (spe_name)
     
    indiv_prob, input_label, prob_res, loc_res = test_step()
    #with open("/home/shared/data4esrd/small_map_data_%d/predict_prob_case_%d_month_%d.csv" % (FLAGS.mon, FLAGS.case, FLAGS.mon), "w") as f:
    #    cnt = 0
    #    print ("LON,LAT,")
    #    for iii, item in enumerate(spe_name):
    #        if iii == 500 - 1:
    #            f.write(str(item))
    #        else:
    #            f.write(str(item) + ",")
    #    f.write("\n")
    #    for loc, prob in zip(loc_res, prob_res):
    #        cnt += 1
    #        f.write(str(float(loc[0])) + "," + str(float(loc[1])) + ",") 
    #        tot = 0
    #        for item in prob:
    #            if tot == 0:
    #                f.write(str(item))
    #            else:
    #                f.write("," + str(item))
    #            tot += 1
    #        f.write("\n")



    """analysis("all", indiv_prob, input_label, True)

    summary = []
    for i in range(FLAGS.r_dim):
        #print i
        sp_indiv_prob = indiv_prob[:,i].reshape(indiv_prob.shape[0],1)
        sp_input_label = input_label[:,i].reshape(input_label.shape[0],1)

        res = analysis(i, sp_indiv_prob, sp_input_label, False)
        summary.append(res)
    summary = np.asarray(summary)
    

    np.save("../data/summary_all",summary) #save the analysis result to ../data/summary_all.npy, which loged the performance indicators of each entity
    """
    
if __name__=='__main__':
    tf.app.run()



