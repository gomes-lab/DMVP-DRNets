import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
FLAGS = tf.app.flags.FLAGS

class MODEL:

    def __init__(self,is_training):

        tf.set_random_seed(19950420)

        r_dim = FLAGS.r_dim
        
        self.input_nlcd = tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.nlcd_dim],name='input_nlcd')
        
        self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.r_dim],name='input_label')

        self.keep_prob = tf.placeholder(tf.float32) #keep probability for the dropout

        weights_regularizer = slim.l2_regularizer(FLAGS.weight_decay)

        ############## compute mu & sigma ###############

        self.fc_1 = slim.fully_connected(self.input_nlcd, 256, weights_regularizer=weights_regularizer, scope='extractor/fc_1')
        self.fc_2 = slim.fully_connected(self.fc_1, 512, weights_regularizer=weights_regularizer, scope='extractor/fc_2')
        #self.fc_3 = slim.fully_connected(self.fc_2, 256, weights_regularizer=weights_regularizer, scope='generator/fc_3')

        #dropout
        #feature1 = slim.dropout(self.fc_3, keep_prob=self.keep_prob, is_training=is_training)

        feature1=self.fc_2              
        
        #compute the mean of the normal random variables 
        self.r_mu = slim.fully_connected(feature1, r_dim, activation_fn=None, weights_regularizer=weights_regularizer, scope='r_mu')
        
        #log the average of the absolute value of means
        #tf.summary.scalar("mu_abs_mean", tf.reduce_mean(tf.abs(self.r_mu)))

        #initialize the square root of the residual covariance matrix 
        self.r_sqrt_sigma=tf.Variable(np.random.uniform(-np.sqrt(6.0/(r_dim+FLAGS.z_dim)), np.sqrt(6.0/(r_dim+FLAGS.z_dim)), (r_dim, FLAGS.z_dim)), dtype=tf.float32, name='r_sqrt_sigma')

        #compute the residual covariance matrix, which is guaranteed to be semi-positive definite
        self.sigma=tf.matmul(self.r_sqrt_sigma, tf.transpose(self.r_sqrt_sigma))


        #tf.summary.scalar("max-cov", tf.reduce_max(tf.abs(self.sigma)))
        #tf.summary.scalar("min-cov", tf.reduce_min(self.sigma))
        #tf.summary.histogram("1D-covariance", tf.reshape(self.sigma,[-1]))

        #self.sqrt_diag=1.0/tf.sqrt(tf.diag_part(self.sigma))
        self.covariance=self.sigma + tf.eye(r_dim)
        self.cov_diag = tf.diag_part(self.covariance)
        #tf.summary.histogram("correlation", tf.reshape(self.correlation,[-1]))
        ############## Sample_r ###############

        self.eps2=tf.constant(1e-6*2.0**(-100), dtype="float64")
        self.eps1=tf.constant(1e-6, dtype="float32")
        self.eps3=1e-30
        ############## alpha tuning##############
        #self.alpha=tf.Variable(10, dtype=tf.float32, name='hyper/alpha')
        #tf.summary.scalar("alpha", self.alpha)
        #self.alpha = tf.constant(1.70169,dtype="float32") #logistic(alphax)~cdf_normal(x)
        #self.beta = tf.constant(1,dtype="float32")

        #noise = tf.random_normal(shape=tf.shape(r_mu))
        #sample_r = r_mu + noise
        #self.sample_nll = tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_r, labels=self.input_label, name='generator/idividual_nll')
        n_sample = FLAGS.n_train_sample
        if (is_training==False):
            n_sample = FLAGS.n_test_sample

        print ("n_sample=",n_sample)

        self.noise = tf.random_normal(shape=[n_sample, tf.shape(self.r_mu)[0], FLAGS.z_dim])
        
        self.B = tf.transpose(self.r_sqrt_sigma)#*self.sqrt_diag
        #tf.summary.histogram("B", tf.reshape(tf.matmul(tf.transpose(self.B), self.B),[-1]))
        
        self.sample_r = tf.tensordot(self.noise, self.B, axes=1)+self.r_mu #tensor: n_sample*n_batch*r_dim
        norm=tf.distributions.Normal(0., 1.)
        E = norm.cdf(self.sample_r)*(1-self.eps1)+self.eps1*0.5

        self.E=E

        #compute the loss for each sample point
        self.sample_nll = tf.negative((tf.log(E)*self.input_label+tf.log(1-E)*(1-self.input_label)), name='sample_nll')
        self.logprob=-tf.reduce_sum(self.sample_nll, axis=2)
        
        #the following computation is designed to avoid the float overflow
        self.maxlogprob=tf.reduce_max(self.logprob, axis=0)
        self.Eprob=tf.reduce_mean(tf.exp(self.logprob-self.maxlogprob), axis=0)
        self.nll_loss=tf.reduce_mean(-tf.log(self.Eprob)-self.maxlogprob)
        
        
        ######### analysis stderr, mean, ratio... ##############
        #a=tf.cast(self.maxlogprob, dtype="float64")
        #b=tf.exp(tf.cast(self.logprob-self.maxlogprob, dtype="float64"))
        #realprob=b
        #stderr=tf.sqrt(tf.reduce_sum( tf.square(b-tf.reduce_mean(b, axis=0) ), axis=0)/(n_sample-1.0))
        #self.std=stderr

        #self.maxstderr=tf.reduce_max(stderr)
        #self.meanstderr=tf.reduce_mean(stderr)
        #self.minstderr=tf.reduce_min(stderr)
        #self.meanprob=tf.reduce_mean(realprob, axis=0)
        #self.maxprob=tf.exp(a)
        #logratio=tf.log(stderr+self.eps3)-tf.log(self.meanprob+self.eps3)

        #tf.summary.histogram("logstderr", tf.log(stderr+self.eps3))
        #tf.summary.scalar("meanP", tf.reduce_mean(self.meanprob))
        #tf.summary.histogram("logratio", logratio)
        
        #tf.summary.histogram("realprob", tf.reshape(realprob,[-1]))
        #self.meanratio=tf.reduce_mean(logratio)
        #self.maxratio=tf.reduce_max(logratio)
        #self.minratio=tf.reduce_min(logratio)
        #tf.summary.scalar("max-logratio", self.maxratio)
        
        #the individual probability of being positive for each label
        self.indiv_prob2 = tf.reduce_mean(E , axis=0, name='individual_prob')

        self.normalized_mu = self.r_mu / tf.sqrt(1e-9 + self.cov_diag)
        self.indiv_prob = norm.cdf(self.normalized_mu)*(1-self.eps1)+0.5*self.eps1

        cross_entropy = (tf.log(self.indiv_prob) * self.input_label + tf.log(1.0 - self.indiv_prob) * (1 - self.input_label))
        self.marginal_loss = -tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))
        
        
        ###### loss ##############
        tf.summary.scalar('nll_loss',self.nll_loss)

        self.l2_loss = tf.add_n(tf.losses.get_regularization_losses())#+FLAGS.weight_decay*tf.nn.l2_loss(self.r_sqrt_sigma)
        tf.summary.scalar('l2_loss',self.l2_loss)
        
        self.total_loss = self.l2_loss + self.nll_loss
        tf.summary.scalar('total_loss',self.total_loss)

        
