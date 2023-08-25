import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import config 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

FLAGS = tf.app.flags.FLAGS
def get_label(data,my_order):

	output = []
	offset = FLAGS.loc_offset + FLAGS.r_offset

	#for i in my_order:
	#output.append(data[i][offset:(offset+FLAGS.r_dim)])
	output = [data[i][offset:(offset+FLAGS.r_dim)] for i in my_order]
	output = np.array(output, dtype="int") 
	return output

def get_nlcd(data,my_order, offset = None):

	output = []
	if (offset == None):
		offset = FLAGS.loc_offset + FLAGS.r_max_dim
		
	#for i in my_order:
	#output.append(data[i][offset:offset + FLAGS.nlcd_dim])
	output = [data[i][offset:offset + FLAGS.nlcd_dim] for i in my_order]
	output = np.array(output, dtype="float32") 
	return output

def get_user(data,my_order):

	output = []
	offset = FLAGS.loc_offset + FLAGS.r_max_dim + FLAGS.nlcd_dim
	for i in my_order:
		output.append(data[i][offset:offset + FLAGS.user_dim])

	output = np.array(output, dtype="float32") 
	return output

def get_loc(data,my_order):

	output = []
	for i in my_order:
		output.append(data[i][:FLAGS.loc_offset])

	output = np.array(output, dtype="float32") 
	return output

######################################################################
from PIL import Image

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def get_jpg_train(images,my_order):

	input_images = []

	for i in my_order:
		img = Image.open('../real_pic/'+images[i])
		img = img.resize((224,224),Image.BILINEAR)
		flip = np.random.randint(2)==1
		if flip:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)

		angle = np.random.randint(4)*90.0
		img = img.rotate(angle,Image.BILINEAR)
		single_image = np.array(img)
		input_images.append(single_image)

	input_images = np.array(input_images)
	input_images = np.split(input_images,3,axis=3)
	means = [_R_MEAN,_G_MEAN,_B_MEAN]
	for j in range(3):
		input_images[j] = input_images[j]-means[j]
	input_images = np.concatenate(input_images,3)
	return input_images

def get_jpg_test(images,my_order):

	input_images = []

	for i in my_order:
		img = Image.open('../real_pic/'+images[i])
		img = img.resize((224,224),Image.BILINEAR)
		single_image = np.array(img)
		input_images.append(single_image)

	input_images = np.array(input_images)
	input_images = np.split(input_images,3,axis=3)
	means = [_R_MEAN,_G_MEAN,_B_MEAN]
	for j in range(3):
		input_images[j] = input_images[j]-means[j]

	input_images = np.concatenate(input_images,3)

	return input_images


