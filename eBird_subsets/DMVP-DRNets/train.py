import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import model
import get_data
import config 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import os, sys
import timeit

from scipy import stats

FLAGS = tf.app.flags.FLAGS

def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

def Species_acc(pred, Y): #the smaller the better
    return np.mean(np.abs(pred - Y))

def Species_Dis(pred, Y): #the larger the better
    res = []
    for i in range(pred.shape[1]):
        try:
            auc = roc_auc_score(Y[:, i] ,pred[:, i])
            res.append(auc)
        except:
            res.append(1.0) #print("AUC nan", i, np.mean(Y[:, i]), np.mean(pred[:, i]))
        
    return np.mean(res)

def Species_Cali(pred, Y):
    res = []
    for j in range(pred.shape[1]):
        p = pred[:, j]
        y = Y[:, j]

        bin1 = np.zeros(10)
        bin2 = np.zeros(10)
        th = np.zeros(10)

        for k in range(10):
            th[k] = np.percentile(p, (k+1)*10)

        for i in range(p.shape[0]):
            for k in range(10):
                if (p[i] <= th[k]):
                    bin1[k] += p[i]
                    bin2[k] += y[i]
                    break

        diff = np.sum(np.abs(bin1 - bin2))
        #print(bin1)
        #print(bin2)
        res.append(diff)
    return np.mean(res)

def Species_Prec(pred, Y): #the smaller the better
    return np.mean(np.sqrt(pred * (1 - pred)))

def Richness_Acc(pred, Y): #the smaller the better
    return np.sqrt(np.mean((np.sum(pred, axis = 1)-np.sum(Y, axis = 1)) ** 2))

def Richness_Dis(pred, Y): #the larger the better
    return stats.spearmanr(np.sum(pred, axis = 1), np.sum(Y, axis = 1))[0]

def Richness_Cali(pred, Y): #the smaller the better
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))
    richness = np.sum(samples, axis = 2) #100, n
    gt_richness = np.sum(Y, axis = 1)

    res = []

    for i in range(pred.shape[0]):
        if (gt_richness[i] <= np.percentile(richness[:, i], 75) and gt_richness[i] >= np.percentile(richness[:, i], 25)):
            res.append(1)
        else:
            res.append(0)
    p = np.mean(res)
    return np.abs(p - 0.5)

def Richness_Prec(pred, Y): #the smaller the better
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))

    return np.mean(np.std(np.sum(samples, axis = 2), axis = 0))

def Beta_SOR(x, y):
    if (np.sum(x * y) == 0 and np.sum(x + y) == 0):
        return 0

    return 1 - 2 * np.sum(x * y)/np.maximum(np.sum(x + y), 1e-9)

def Beta_SIM(x, y):
    if (np.sum(x * y) == 0 and np.minimum(np.sum(x), np.sum(y)) == 0):
        return 0
    return 1 - np.sum(x * y)/np.maximum(np.minimum(np.sum(x), np.sum(y)), 1e-9)

def Beta_NES(x, y):
    return Beta_SOR(x, y) - Beta_SIM(x, y)

def get_dissim(pred, Y):
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))

    pairs = []
    N = 300
    for i in range(N):
        x = np.random.randint(pred.shape[0])
        y = np.random.randint(pred.shape[0])
        pairs.append([x, y])


    SOR = np.zeros((N, 100))
    SIM = np.zeros((N, 100))
    NES = np.zeros((N, 100))

    gt_SOR = []
    gt_SIM = []
    gt_NES = []
    for i in range(N):
        x, y = pairs[i]
        for j in range(100):
            SOR[i][j] = Beta_SOR(samples[j][x], samples[j][y])
            SIM[i][j] = Beta_SIM(samples[j][x], samples[j][y])
            NES[i][j] = Beta_NES(samples[j][x], samples[j][y])

        gt_SOR.append(Beta_SOR(Y[x], Y[y]))
        gt_SIM.append(Beta_SIM(Y[x], Y[y]))
        gt_NES.append(Beta_NES(Y[x], Y[y]))
    return SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES

def Community_Acc(pred, Y): #the smaller the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return np.sqrt(np.mean((np.mean(SOR, axis = 1) - gt_SOR)**2)),\
    np.sqrt(np.mean((np.mean(SIM, axis = 1) - gt_SIM)**2)),\
    np.sqrt(np.mean((np.mean(NES, axis = 1) - gt_NES)**2))

def Community_Dis(pred, Y): #the larger the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return stats.spearmanr(np.mean(SOR, axis = 1), gt_SOR)[0],\
    stats.spearmanr(np.mean(SIM, axis = 1), gt_SIM)[0],\
    stats.spearmanr(np.mean(NES, axis = 1), gt_NES)[0]

def Community_Cali(pred, Y):
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    tmp1 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(SOR, 25, axis = 1), gt_SOR),\
     np.greater_equal(np.percentile(SOR, 75, axis = 1),gt_SOR)).astype("float")) - 0.5)

    tmp2 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(SIM, 25, axis = 1), gt_SIM),\
     np.greater_equal(np.percentile(SIM, 75, axis = 1),gt_SIM)).astype("float")) - 0.5)

    tmp3 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(NES, 25, axis = 1), gt_NES),\
     np.greater_equal(np.percentile(NES, 75, axis = 1),gt_NES)).astype("float")) - 0.5)

    return tmp1, tmp2, tmp3

def Community_Prec(pred, Y): #the smaller the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return np.mean(np.std(SOR, axis = 1)), \
    np.mean(np.std(SIM, axis = 1)), \
    np.mean(np.std(NES, axis = 1))


def validation_step(sess, hg, DL, merged_summary, summary_writer, valid_idx, epoch, metrics, metric_names):

    print('Validating...')

    valid_log = get_data.Log()


    batch_size = FLAGS.batch_size

    Ys = []
    preds = []
    for i in range((len(valid_idx)-1)//batch_size + 1):

        start = batch_size*i
        end = min(batch_size*(i+1), len(valid_idx))

        input_X = DL.get_X(valid_idx[start:end])
        input_Y = DL.get_Y(valid_idx[start:end])

        feed_dict={}
        feed_dict[hg.input_X]=input_X
        feed_dict[hg.input_Y]=input_Y
        feed_dict[hg.keep_prob]=1.0

        nll_loss, l2_loss, total_loss, marginal_loss, indiv_prob = sess.run([hg.nll_loss, hg.l2_loss, hg.total_loss, hg.marginal_loss, hg.indiv_prob],feed_dict)
        
        Ys.append(input_Y)
        preds.append(indiv_prob)

        valid_log.add("marginal_loss", marginal_loss)
        valid_log.add("nll_loss", nll_loss)


    results = valid_log.get_means(valid_log.get_var_names(), clean = False)
    print("validation-%d"%epoch, end = " ")
    for (name, avg) in results:
        print("%s: %.3f"%(name, avg), end = " ")
    print()

    Ys = np.concatenate(Ys, axis = 0)
    preds = np.concatenate(preds, axis = 0)

    summary_writer.add_summary(MakeSummary('validation/nll_loss', valid_log.get_mean("nll_loss")), epoch)
    summary_writer.add_summary(MakeSummary('validation/marginal_loss', valid_log.get_mean("marginal_loss")), epoch)

    #return Species_acc(preds, Ys), preds, Ys
    return valid_log.get_mean("nll_loss"), preds, Ys

def main(_):

    start_time = timeit.default_timer()
    print('reading npy...')
    np.random.seed(19950420) # set the random seed of numpy
    DL = get_data.Data_loader(sys.argv[1], sys.argv[2], use_S = True)
    train_log = get_data.Log()

    train_idx, test_idx, n_feature, n_classes = DL.get_indices()
    
    np.random.shuffle(train_idx)
    
    M = len(train_idx)//10
    valid_idx = train_idx[:M] 
    train_idx = train_idx[M:]

    print("n_feature", n_feature, "n_classes", n_classes)

    one_epoch_iter = len(train_idx) // FLAGS.batch_size # compute the number of iterations in each epoch

    print('reading completed')


    metrics = [Species_acc, Species_Dis, Species_Cali, Species_Prec, \
              Richness_Acc, Richness_Dis, Richness_Cali, Richness_Prec, 
              Community_Acc, Community_Dis, Community_Cali, Community_Prec]

    metric_names = ["Species_acc", "Species_Dis", "Species_Cali", "Species_Prec", \
                 "Richness_Acc", "Richness_Dis", "Richness_Cali", "Richness_Prec", \
                 "Community_Acc", "Community_Dis", "Community_Cali", "Community_Prec"]
    # config the tensorflow
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)

    print('showing the parameters...\n')

    #parameterList = FLAGS.__dict__['__flags'].items()
    #parameterList = sorted(parameterList)

    # print all the hyper-parameters in the current training

    print('building network...')

    #building the model 
    hg = model.MODEL(is_training=True, n_feature = n_feature, n_classes = n_classes)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, (1.0/FLAGS.lr_decay_times)*(FLAGS.max_epoch*one_epoch_iter), FLAGS.lr_decay_ratio, staircase=True)

    #log the learning rate 
    tf.summary.scalar('learning_rate', learning_rate)

    #use the Adam optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate)

    #set training update ops/backpropagation
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(hg.total_loss, global_step = global_step)

    merged_summary = tf.summary.merge_all() # gather all summary nodes together
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph) #initialize the summary writer

    sess.run(tf.global_variables_initializer()) # initialize the global variables in tensorflow
    saver = tf.train.Saver(max_to_keep=FLAGS.max_keep) #initializae the model saver

    print('building finished')

    #initialize several 
    best_loss = 1e10 
    best_iter = 0
    
    drop_cnt = 0
    max_epoch = FLAGS.max_epoch
    #if (len(train_idx) < 5000): # train more epochs for small datasets
    #    max_epoch *= 5
    for epoch in range(max_epoch):
        
        print('epoch '+str(epoch+1)+' starts!')

        np.random.shuffle(train_idx) # random shuffle the training indices  
        
        for i in range(len(train_idx)//FLAGS.batch_size):
            
            start = i*FLAGS.batch_size
            end = (i+1)*FLAGS.batch_size

            input_X = DL.get_X(train_idx[start:end]) # get the X features 
            input_Y = DL.get_Y(train_idx[start:end]) # get the prediction Ys 

            #train the model for one step and log the training loss
            feed_dict={}
            feed_dict[hg.input_X]=input_X
            feed_dict[hg.input_Y]=input_Y
            feed_dict[hg.keep_prob]=0.5

            temp, current_step, nll_loss, l2_loss, total_loss, marignal_loss, summary, indiv_prob= \
            sess.run([train_op, global_step, hg.nll_loss, hg.l2_loss, hg.total_loss, hg.marginal_loss, \
            merged_summary, hg.indiv_prob], feed_dict)

            train_log.add("marignal_loss", marignal_loss)
            train_log.add("nll_loss", nll_loss)

            if (current_step + 1) % FLAGS.check_freq==0: #summarize the current training status and print them out
                results = train_log.get_means(train_log.get_var_names())
                print("iteration-%d"%current_step, end = " ")
                for (name, avg) in results:
                    print("%s: %.3f"%(name, avg), end = " ")
                print()
            summary_writer.add_summary(summary, current_step)
    

        if (epoch + 1) % FLAGS.save_epoch == 0: #exam the model on validation set

            #exam the model on validation set
            current_loss, preds, Ys = validation_step(sess, hg, DL, merged_summary, summary_writer, valid_idx, epoch, metrics, metric_names)

            if (current_loss < best_loss): # no model selection, run to a fixed epoch 

                print('current loss:%.10f  which is better than the previous best one!!!'%current_loss)

                best_loss = current_loss
                best_iter = epoch
                test_loss, preds, Ys = validation_step(sess, hg, DL, merged_summary, summary_writer, valid_idx, epoch, metrics, metric_names)
                best_res = [preds, Ys]
                drop_cnt = 0
            else:
                drop_cnt += 1
            
            if (drop_cnt > 10):
                break



    print('training completed !')
    print('the best loss on validation is '+str(best_loss))
    print('the best checkpoint is '+str(best_iter))

    end_time = timeit.default_timer()
    print("Running time:", end_time - start_time)

    preds, Ys = best_res
    Res = []
    for i in range(len(metrics)):
        f = metrics[i]
        name = metric_names[i]
        res = (name, f(preds, Ys))
        print(res)
        if (isinstance(res[1], tuple)):
            for x in res[1]:
                Res.append(x)
        else:
            Res.append(res[1])

    np.save("results/%s_%s"%(sys.argv[1], sys.argv[2]), Res)


if __name__=='__main__':
    tf.app.run()
