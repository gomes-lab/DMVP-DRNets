import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import datetime
import model
import get_data
import config 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

FLAGS = tf.app.flags.FLAGS

def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

def train_step(sess, hg, merged_summary, summary_writer, input_label, input_nlcd, train_op, global_step):

    feed_dict={}
    feed_dict[hg.input_nlcd]=input_nlcd
    feed_dict[hg.input_label]=input_label
    feed_dict[hg.keep_prob]=0.8

    temp, step, nll_loss, marginal_loss, l2_loss, total_loss, summary, indiv_prob= \
    sess.run([train_op, global_step, hg.nll_loss, hg.marginal_loss, hg.l2_loss, hg.total_loss,\
     merged_summary, hg.indiv_prob], feed_dict)

    time_str = datetime.datetime.now().isoformat()
    summary_writer.add_summary(summary,step)

    return indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss

def validation_step(sess, hg, data, merged_summary, summary_writer, valid_idx, global_step):

    print ('Validating...')

    all_nll_loss = 0
    all_l2_loss = 0
    all_total_loss = 0
    all_marginal_loss = 0

    all_indiv_prob = []
    all_label = []

    real_batch_size=min(FLAGS.batch_size, len(valid_idx))
    for i in range(int( (len(valid_idx)-1)/real_batch_size )+1):

        start = real_batch_size*i
        end = min(real_batch_size*(i+1), len(valid_idx))

        input_nlcd = get_data.get_nlcd(data,valid_idx[start:end])
        input_label = get_data.get_label(data,valid_idx[start:end])

        feed_dict={}
        feed_dict[hg.input_nlcd]=input_nlcd
        feed_dict[hg.input_label]=input_label
        feed_dict[hg.keep_prob]=1.0


        nll_loss, marginal_loss, l2_loss, total_loss, indiv_prob = sess.run([hg.nll_loss, hg.marginal_loss, hg.l2_loss, hg.total_loss, hg.indiv_prob],feed_dict)
    
        all_nll_loss += nll_loss*(end-start)
        all_l2_loss += l2_loss*(end-start)
        all_total_loss += total_loss*(end-start)
        all_marginal_loss += marginal_loss * (end - start)
    
        for i in indiv_prob:
            all_indiv_prob.append(i)
        for i in input_label:
            all_label.append(i)

    all_indiv_prob = np.array(all_indiv_prob)
    all_label = np.array(all_label)

    #auc = roc_auc_score(all_label,all_indiv_prob)

    nll_loss = all_nll_loss/len(valid_idx)
    l2_loss = all_l2_loss/len(valid_idx)
    total_loss = all_total_loss/len(valid_idx)
    marginal_loss = all_marginal_loss / len(valid_idx)

    all_indiv_prob=np.reshape(all_indiv_prob,(-1))
    all_label=np.reshape(all_label,(-1))
    ap = average_precision_score(all_label,all_indiv_prob)

    time_str = datetime.datetime.now().isoformat()

    #print ("validation results: %s\tauc=%.6f\tap=%.6f\tnll_loss=%.6f\tmarginal_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f" % (time_str, auc, ap, nll_loss, marginal_loss, l2_loss, total_loss))
    print ("validation results: %s\tap=%.6f\tnll_loss=%.6f\tmarginal_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f" % (time_str, ap, nll_loss, marginal_loss, l2_loss, total_loss))
                    
    current_step = sess.run(global_step) #get the value of global_step
    #summary_writer.add_summary(MakeSummary('validation/auc',auc),current_step)
    summary_writer.add_summary(MakeSummary('validation/ap',ap),current_step)
    summary_writer.add_summary(MakeSummary('validation/nll_loss',nll_loss),current_step)

    return nll_loss

def main(_):

    st_time = time.time()
    print ('reading npy...')
    np.random.seed(19950420) # set the random seed of numpy 
    data = np.load(FLAGS.data_dir) #load data from the data_dir
    #srd = np.load(FLAGS.srd_dir)
    train_idx = np.load(FLAGS.train_idx) #load the indices of the training set
    valid_idx = np.load(FLAGS.valid_idx) #load the indices of the validation set
    labels = get_data.get_label(data, train_idx) #load the labels of the training set

    print ("positive label rate:", np.mean(labels)) #print the rate of the positive labels in the training set

    one_epoch_iter = len(train_idx) / FLAGS.batch_size # compute the number of iterations in each epoch

    print ('reading completed')

    # config the tensorflow
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)

    print ('showing the parameters...\n')

    #parameterList = FLAGS.__dict__['__flags'].items()
    #parameterList = sorted(parameterList)

    # print all the hyper-parameters in the current training
    #for (key, value) in FLAGS.__dict__['__flags'].items():
    #   print ("%s\t%s"%(key, value))
    #print ("\n")


    print ('building network...')

    #building the model 
    hg = model.MODEL(is_training=True)

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

    print ('building finished')

    #initialize several 
    best_loss = 1e10 
    best_iter = 0

    smooth_nll_loss=0.0
    smooth_l2_loss=0.0
    smooth_total_loss=0.0

    temp_label=[]   
    temp_indiv_prob=[]

    for one_epoch in range(FLAGS.max_epoch):
        
        print('epoch '+str(one_epoch+1)+' starts!')

        np.random.shuffle(train_idx) # random shuffle the training indices  
        
        for i in range(int(len(train_idx)/float(FLAGS.batch_size))):
            
            start = i*FLAGS.batch_size
            end = (i+1)*FLAGS.batch_size

            input_nlcd = get_data.get_nlcd(data,train_idx[start:end]) # get the NLCD features 
            input_label = get_data.get_label(data,train_idx[start:end]) # get the prediction labels 

            #train the model for one step and log the training loss
            indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss = train_step(sess, hg, merged_summary, summary_writer, input_label,input_nlcd, train_op, global_step)
            
            smooth_nll_loss += nll_loss
            smooth_l2_loss += l2_loss
            smooth_total_loss += total_loss
            
            temp_label.append(input_label) #log the labels
            temp_indiv_prob.append(indiv_prob) #log the individual prediction of the probability on each label

            current_step = sess.run(global_step) #get the value of global_step

            if current_step%FLAGS.check_freq==0: #summarize the current training status and print them out

                nll_loss = smooth_nll_loss / float(FLAGS.check_freq)
                l2_loss = smooth_l2_loss / float(FLAGS.check_freq)
                total_loss = smooth_total_loss / float(FLAGS.check_freq)
                
                temp_indiv_prob = np.reshape(np.array(temp_indiv_prob),(-1))
                temp_label = np.reshape(np.array(temp_label),(-1))

                ap = average_precision_score(temp_label,temp_indiv_prob) #compute the AP indicator 

                temp_indiv_prob = np.reshape(temp_indiv_prob,(-1,FLAGS.r_dim))
                temp_label = np.reshape(temp_label,(-1,FLAGS.r_dim))

                try:
                    pass
                    #print (temp_label)
                    #auc = roc_auc_score(temp_label,temp_indiv_prob) # compute the auc indicator 

                except ValueError:
                    print ('y true error for auc')

                else:
                    time_str = datetime.datetime.now().isoformat()
                    #print out the real-time status of the model  
                    #print ("%s\tstep=%d\tauc=%.6f\tap=%.6f\tnll_loss=%.6f\tmarginal_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f" % (time_str, current_step, auc, ap, nll_loss, marginal_loss, l2_loss, total_loss))
                    print ("%s\tstep=%d\tap=%.6f\tnll_loss=%.6f\tmarginal_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f" % (time_str, current_step, ap, nll_loss, marginal_loss, l2_loss, total_loss))
                    #summary_writer.add_summary(MakeSummary('train/auc',auc),current_step)
                    #summary_writer.add_summary(MakeSummary('train/ap',ap),current_step)

                temp_indiv_prob=[]
                temp_label=[]

                smooth_nll_loss = 0
                smooth_l2_loss = 0
                smooth_total_loss = 0

            if current_step % int(one_epoch_iter*FLAGS.save_epoch)==0: #exam the model on validation set

                #exam the model on validation set
                current_loss = validation_step(sess, hg, data, merged_summary, summary_writer, valid_idx, global_step)

                if current_loss < best_loss: # find a better model than the last checkpoint

                    print ('current loss:%.10f  which is better than the previous best one!!!'%current_loss)

                    best_loss = current_loss
                    best_iter = current_step

                    print ('saving model')
                    saved_model_path = saver.save(sess,FLAGS.model_dir+'model',global_step=current_step)

                    print ('have saved model to ', saved_model_path)

                    #write the best checkpoint number back to the config.py file
                    #configFile=open(FLAGS.config_dir, "r")
                    #content=[line.strip("\n") for line in configFile]
                    #configFile.close()

                    #for i in range(len(content)):
                    #    if ("checkpoint_path" in content[i]):
                    #        content[i]="tf.app.flags.DEFINE_string('checkpoint_path', './model/model-%d','The path to a checkpoint from which to fine-tune.')"%best_iter
                    #
                    #configFile=open(FLAGS.config_dir, "w")
                    #for line in content:
                    #    configFile.write(line+"\n")
                    #configFile.close()

    print ('training completed !')
    print ('the best loss on validation is '+str(best_loss))
    print ('the best checkpoint is '+str(best_iter))
    ed_time = time.time()
    print ("running time: ", ed_time - st_time)   

if __name__=='__main__':
    tf.app.run()
    
