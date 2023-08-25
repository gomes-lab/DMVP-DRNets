import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', './model/','path to store the checkpoints of the model')
tf.app.flags.DEFINE_string('summary_dir', './summary','path to store analysis summaries used for tensorboard')
tf.app.flags.DEFINE_string('config_dir', './config.py','path to config.py')
tf.app.flags.DEFINE_string('checkpoint_path', './model/model-43884','The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string('visual_dir', './visualization/','path to store visualization codes and data')

tf.app.flags.DEFINE_string('data_dir', '../data/bird.npy','The path of input observation data')
#tf.app.flags.DEFINE_string('srd_dir', '../data/loc_ny_nlcd2006.npy','The path of input srd data')
tf.app.flags.DEFINE_string('train_idx', '../data/bird_train_idx.npy','The path of training data index')
tf.app.flags.DEFINE_string('valid_idx', '../data/bird_val_idx.npy','The path of validation data index')
tf.app.flags.DEFINE_string('test_idx', '../data/bird_test_idx.npy','The path of testing data index')

tf.app.flags.DEFINE_integer('batch_size', 128, 'the number of data points in one minibatch') #128
tf.app.flags.DEFINE_integer('testing_size', 128, 'the number of data points in one testing or validation batch') #128
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate')
#tf.app.flags.DEFINE_float('pred_lr', 1, 'the learning rate for predictor')
tf.app.flags.DEFINE_integer('max_epoch', 200, 'max epoch to train')
tf.app.flags.DEFINE_float('weight_decay', 0.00001, 'weight decay rate')
tf.app.flags.DEFINE_float('threshold', 0.5, 'The probability threshold for the prediction')
tf.app.flags.DEFINE_float('lr_decay_ratio', 0.5, 'The decay ratio of learning rate')
tf.app.flags.DEFINE_float('lr_decay_times', 3.0, 'How many times does learning rate decay')
tf.app.flags.DEFINE_integer('n_test_sample', 10000, 'The sampling times for the testing')
tf.app.flags.DEFINE_integer('n_train_sample', 100, 'The sampling times for the training') #100


tf.app.flags.DEFINE_integer('z_dim', 20, 'z dimention: the number of the independent normal random variables in DMSE \
    / the rank of the residual covariance matrix')

tf.app.flags.DEFINE_integer('r_offset', 0, 'the offset of labels')
tf.app.flags.DEFINE_integer('r_dim', 100, 'the number of labels in current training') #100
tf.app.flags.DEFINE_integer('r_max_dim', 100, 'max number of all labels in the dataset')

tf.app.flags.DEFINE_integer('loc_offset', 2, 'the offset caused by location features') #100
tf.app.flags.DEFINE_integer('nlcd_dim', 15, 'the dimensionality of the NLCD features ')
#tf.app.flags.DEFINE_integer('user_dim', 6, 'the dimensionality of the user-features')

tf.app.flags.DEFINE_float('save_epoch', 2.0, 'epochs to save the checkpoint of the model')
tf.app.flags.DEFINE_integer('max_keep', 3, 'maximum number of saved model')
tf.app.flags.DEFINE_integer('check_freq', 10, 'checking frequency')



