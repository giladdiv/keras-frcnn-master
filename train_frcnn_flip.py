from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os
from keras.losses import mean_squared_error
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input,Lambda,Flatten,Activation
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import resnet_FC_trip as nn
import keras_frcnn.roi_helpers as roi_helpers
import keras_frcnn.img_helper as img_helpers
from keras.utils import generic_utils
from keras_frcnn.SliceTensor import SliceTensor
import imageio
from PIL import Image
import tensorflow as tf
from keras.utils import plot_model
import copy
from test_view_func import *

#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


sys.setrecursionlimit(40000)

parser = OptionParser()
base_path = os.getcwd()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.",default='./VOCdevkit')
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc"),
parser.add_option("-n", "--num_rois", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--num_epochs", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_flip_nosub.hdf5')

parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.",default ='./weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

data_type = 'real'
if data_type == 'real':
	parser.add_option("--input_train_file", dest="input_train_file", help="if there is a pickle file for train data.",default='pickle_data/train_data_Wflip_all.pickle' )
else:
	parser.add_option("--input_train_file", dest="input_train_file", help="if there is a pickle file for train data.",
					  default='pickle_data/syn_by_obj_data.pickle')

(options, args) = parser.parse_args()

cls_num = 21
data_len = 360
fc_size = 1024
fc_size = 360
last_layer = (options.num_rois,cls_num*data_len)

def create_mat_siam(label=7,cls_num=21):
    mat = np.zeros([data_len *cls_num,data_len ],dtype=np.float32)
    for ii in range(data_len ):
		mat[data_len  * label + ii, ii] = 1
    return mat


def mat_lambda(vects):
	# https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
	x = vects
	x= K.reshape(x,[-1,last_layer[1]])
	h =tf.matmul(x,
			  K.constant(create_mat_siam(label=7, cls_num=cls_num), shape=[data_len * cls_num, data_len], dtype=tf.float32))
	h =K.reshape(h,[-1,last_layer[0],data_len])
	return h

def mat_lambda_output_shape(shapes):
	shape1 = shapes
	return(shape1[0],last_layer[0],data_len)

def cosine_distance(vects):
	x, y = vects
	# x = tf.Print(x,[tf.shape(x)])
	dis = K.sum(x*y,axis=2)
	return K.reshape(dis,shape=(-1,32,1))


def cosine_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],shape1[1],1)


def create_mat_flip():
	mat = np.zeros([fc_size,fc_size],dtype=np.float32)
	for ii in range(fc_size):
		mat[fc_size-(ii+1), ii] = 1
	return mat

def create_mat_flip_old():
	mat = np.zeros([data_len,data_len],dtype=np.float32)
	for ii in range(data_len ):
		if ii == 0:
			mat[0,0] = 1
		else:
			mat[data_len-ii, ii] = 1
	return mat


def mat_flip_lambda(vects):
	# https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
	x = vects
	x= K.reshape(x,[-1,fc_size])
	h =tf.matmul(x,
			  K.constant(create_mat_flip(), shape=[fc_size, fc_size], dtype=tf.float32))
	h =K.reshape(h,[-1,last_layer[0],fc_size])
	return h

def mat_flip_lambda_old(vects):
	# https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
	x = vects
	x= K.reshape(x,[-1,fc_size])
	h =tf.matmul(x,
			  K.constant(create_mat_flip_old(), shape=[fc_size, fc_size], dtype=tf.float32))
	h =K.reshape(h,[-1,last_layer[0],fc_size])
	return h

def mat_lambda_flip_output_shape(shapes):
	shape1 = shapes
	return(shape1[0],last_layer[0],fc_size)

def l2_layer(vects):
	x = vects
	return K.sqrt(K.sum(x**2,axis=2))

def l2_layer_output_shape(shapes):
	shape1 = shapes
	return(shape1[0],1)


def trip_layer(vects):
	dp,dm = vects
	return K.l2_normalize(K.concatenate([dp,dm],axis=-1),axis=1)

def euclidean_distance(vects):
	x, y = vects
	# x = tf.Print(x,[tf.shape(x)])
	return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 5
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def slice_vec(vects):
	x,y = vects
	roi_num =32
	num_classes =21
	pred = x
	labels_mask = y[0, :, 360:-1]
	labels = tf.to_int32(y[0, :, -1])

	## find the indicies of the bg
	bg = tf.constant(num_classes - 1, dtype=tf.int32)
	indices = tf.where(tf.not_equal(labels, bg))

	zero = tf.constant(0, dtype=tf.float32)
	for i in range(roi_num):
		indices_mask = tf.where(tf.not_equal(labels_mask[i, :], zero))
		indices_mask = tf.reshape(indices_mask, [-1])
		# indices_mask = tf.Print(indices_mask, [i])
		# indices_mask = tf.Print(indices_mask, [indices_mask])
		if i == 0:
			fc_az_l = tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])
		else:
			fc_az_l = tf.concat(axis=0, values=[fc_az_l, tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])])

	return fc_az_l


def slice_vec_output_shape(shapes):
    shape1, shape2 = shapes
    return (32, 360)


if not options.train_path:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
	from keras_frcnn.pascal3D_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
draw_flag = False
C = config.Config()
C.num_classes = 21
C.num_rois = int(options.num_rois)
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
temp_ind = C.model_path.index(".hdf5")
C.model_path_epoch =C.model_path[:temp_ind]+'_epoch'+C.model_path[temp_ind:]
C.weight_name = os.path.splitext(os.path.basename(options.output_weight_path))[0]

if options.input_weight_path:
	C.base_net_weights = options.input_weight_path

## read the training data from pickle file or from annotations
if os.path.exists(options.input_train_file):
	with open(options.input_train_file) as f:
		t1=time.time()
		all_imgs, classes_count,_ = pickle.load(f)
		t2 = time.time()
		print('Loading data took {} sec'.format(t2-t1))
else:
	all_imgs, classes_count,_ = get_data(options.train_path)
##

## read sorted data
# sorted_path = options.input_train_file
# tmp_ind = sorted_path.index('.pickle')
# sorted_path = sorted_path[:tmp_ind]+"_sorted_Angles"+sorted_path[tmp_ind:]
# if os.path.exists(sorted_path):
# 	print("loading sorted data")
# 	with open(sorted_path) as f:
# 		trip_data = pickle.load(f)


if 'bg' not in classes_count:
	classes_count['bg'] = 0
	# class_mapping['bg'] = len(class_mapping)

class_mapping = {'sheep': 5, 'bottle': 8, 'horse': 15, 'bg': 20, 'bicycle': 17, 'motorbike': 16, 'cow': 12, 'sofa': 14, 'dog': 0, 'bus': 11, 'cat': 10, 'person': 6, 'train': 4, 'diningtable': 13, 'aeroplane': 19, 'car': 1, 'pottedplant': 2, 'tvmonitor': 7, 'chair': 9, 'bird': 3, 'boat': 18}
C.class_mapping = class_mapping
trip_cls = ['aeroplane','bicycle','bus','car','chair','diningtable','motorbike','sofa','train', 'tvmonitor']

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs]
# val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
# print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='test',create_flip=True)
# data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)
	# input_shape_img_siam = (600, 1066, 3)
	input_shape_img_siam = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))
labels_input = Input(shape=(None, data_len*(len(class_mapping)+1)+1))
feature_input = Input(shape=(None,None,1024))

# cls_input = tf.placeholder(shape=(None,1),dtype=tf.int32)
## siam input placeholders
img_input_ref = Input(shape=input_shape_img_siam)
roi_input_ref = Input(shape=(None, 4))
feature_input_ref = Input(shape=(None,None,1024))


img_input_flip = Input(shape=input_shape_img_siam)
roi_input_flip = Input(shape=(None, 4))
feature_input_flip = Input(shape=(None,None,1024))



optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
optimizer_trip = Adam(lr=1e-5)
rms = RMSprop()

## siam network part
C.siam_iter_frequancy = 6
weight_path_init = os.path.join(base_path, 'models/model_FC_init.hdf5')
# weight_path_tmp = os.path.join(base_path, 'tmp_weights.hdf5')
weight_path_tmp = os.path.join(base_path, 'models/model_frcnn_siam_tmp.hdf5')
NumOfCls = len(class_mapping)

def build_models(weight_path,init_models = False,train_view_only = False,create_siam = False):
	##
	if train_view_only:
		trainable_cls = False
		trainable_view =False
	else:
		trainable_cls = False
		trainable_view =  False
	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable= trainable_cls)

	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn = nn.rpn(shared_layers, num_anchors)


	# classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable_cls=trainable_cls,trainable_view=trainable_view)
	classifier,inner_layer = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=C.num_classes, trainable_cls=trainable_cls,trainable_view=trainable_view)


	# L2 normalization for inner layer
	inner_layer = Lambda(lambda x: tf.nn.l2_normalize(x,dim =2))(inner_layer)

	model_rpn = Model(img_input, rpn[:2])

	model_classifier = Model([img_input, roi_input], classifier)
	# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
	model_all = Model([img_input, roi_input], rpn[:2] + classifier)

	if init_models:
		try:
			print('loading weights from {}'.format(C.base_net_weights))
			model_rpn.load_weights(C.base_net_weights, by_name=True)
			model_classifier.load_weights(C.base_net_weights, by_name=True)
		except:
			print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
				'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
				'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
			))

	## load pre-trained net


	# roi_helpers.compere_weights(model_classifier.get_weights(),model_rpn.get_weights(),0,0)
	model_rpn.load_weights(weight_path, by_name=True)
	model_classifier.load_weights(weight_path, by_name=True)


	model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
	##no weights
	model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(C.num_classes-1),losses.class_loss_view_weight(C.num_classes,roi_num=C.num_rois)], metrics=['accuracy'])
	## with weights
	# model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1),losses.class_loss_view_weight(len(classes_count),roi_num=C.num_rois)], metrics=['accuracy'])

	model_all.compile(optimizer='sgd', loss='mae')

	if create_siam:
		model_view_only = Model([img_input, roi_input], classifier[2])
		model_inner = Model([img_input, roi_input],inner_layer)

		## use the feature map after rpn,train only the view module
		inner_ref = model_inner([img_input_ref,roi_input_ref])
		cls_ref = model_view_only([img_input_ref,roi_input_ref])
		inner_flip = model_inner([img_input_flip,roi_input_flip])
		cls_flip = model_view_only([img_input_flip,roi_input_flip])

		##### first way, flip the feature layer
		# inner_flip = Lambda(mat_flip_lambda,mat_lambda_flip_output_shape)(inner_flip)
		# flat_ref = Flatten()(inner_ref)
		# flat_flip = Flatten()(inner_flip)
        #
		# sub = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([flat_ref,flat_flip])
		# sub = Lambda(lambda x: x * .5)(sub)
        #
		# model_flip = Model([img_input_ref, roi_input_ref, img_input_flip, roi_input_flip], [cls_ref,cls_flip,sub])
		# model_flip.compile(optimizer=optimizer_trip, loss=[losses.class_loss_view_weight(len(classes_count),C.num_rois),losses.class_loss_view_weight(len(classes_count),C.num_rois),'mse'])


		##### second way, flip the last layer
		view_non_flip_before = SliceTensor(len(class_mapping),C.num_rois)([cls_ref,labels_input])
		view_flip = SliceTensor(len(class_mapping),C.num_rois)([cls_flip,labels_input])
		# view_flip = K.expand_dims(view_flip,axis=0)
		view_flip_before = Lambda(mat_flip_lambda_old,output_shape=mat_lambda_flip_output_shape)(view_flip)

		view_flip = Activation('softmax')(view_flip_before)
		view_non_flip = Activation('softmax')(view_non_flip_before)

		flat_flip = Flatten()(view_flip)
		flat_non_flip = Flatten()(view_non_flip)

		distance_flip = Lambda(euclidean_distance,
				output_shape=eucl_dist_output_shape)([flat_non_flip, flat_flip])

		distance_flip = Lambda(cosine_distance,
				   output_shape=cosine_dist_output_shape)([view_non_flip_before, view_flip_before])

		# model_flip = Model([img_input_ref, roi_input_ref, img_input_flip, roi_input_flip,labels_input], [cls_ref,cls_flip,distance_flip])
		# model_flip.compile(optimizer=optimizer_trip, loss=[losses.class_loss_view_weight(len(classes_count),C.num_rois),losses.class_loss_view_weight(len(classes_count),C.num_rois),'mse'])


		model_flip = Model([img_input_ref, roi_input_ref, img_input_flip, roi_input_flip,labels_input], [cls_ref,cls_flip])
		model_flip.compile(optimizer=optimizer_trip, loss=[losses.class_loss_view_weight(len(classes_count),C.num_rois),losses.class_loss_view_weight(len(classes_count),C.num_rois)])

		return model_rpn, model_classifier, model_all, model_inner, model_flip,model_view_only
	else:
		return  model_rpn,model_classifier,model_all




## get layers
# view_only_layers= [i for i,x in enumerate(model_classifier.layers) if 'view' in x.name]
# classifier_not_trainable_layers = [i for i,x in enumerate(model_classifier.layers) if x.trainable == False]
#
# for l in model_classifier.layers:
# 	l.trainable = False


## video loading
model_rpn,model_classifier,model_all = build_models(weight_path=weight_path_init,init_models=True)
# view_only_layers= [i for i,x in enumerate(model_classifier.layers) if 'view' in x.name]
# jj = 140
# rep = 0
# for ii in view_only_layers[:-1]:
# 	flag_view = True
# 	while flag_view:
# 		if model_classifier.layers[ii].name.replace('_view_','5') == model_classifier.layers[jj].name:
# 			model_classifier.layers[ii].set_weights(model_classifier.layers[jj].get_weights())
# 			flag_view = False
# 			rep +=1
# 		jj+=1

model_all.save_weights(weight_path_tmp)
_,_,_,model_inner,model_flip,model_view_only = build_models(weight_path = weight_path_tmp, init_models = True,create_siam=True, train_view_only = True)
## get layers
# test_view_func(C, model_rpn, model_classifier)
# classifier_not_trainable_layers = [i for i,x in enumerate(model_classifier.layers) if x.trainable == False]


epoch_length = 1000
num_epochs = int(options.num_epochs)
iter_num = 0
countNum = 0

MAP = 0
best_succ = 0
best_succ_epoch = 0
epoch_save_num = 5
succ_vec= np.zeros([1,int(np.ceil(float(num_epochs)/float(epoch_save_num)))])

losses = np.zeros((epoch_length, 6))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

def prep_siam(img,C):
	X,ratio= img_helpers.format_img(img,C)
	X = np.transpose(X, (0, 2, 3, 1))
	P_rpn = model_rpn.predict_on_batch(X)
	R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,
								 max_boxes=300)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]
	return X,R,ratio


# func_k = K.function([img_input_ref, roi_input_ref, img_input_flip, roi_input_flip,labels_input],[model_flip.layers[6].output,model_flip.layers[7].output])

for epoch_num in range(num_epochs):

	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num, num_epochs))

	while True:
		# t_start = time.time()
		try:
			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			X, Y, img_data,X_flip, Y_flip, img_data_flip = next(data_gen_train)

			## flip can work only if a single object is present
			if len(img_data['bboxes'])>1:
				continue
			# loss_rpn = model_rpn.train_on_batch(X, Y)

			### deal with regular image
			P_rpn = model_rpn.predict_on_batch(X)

			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, Y_view = roi_helpers.calc_iou_new(R,img_data, C, class_mapping)


			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue

			pos_samples = np.where(Y_view[0, :, -1] != 20)
			sel_samples_regular = pos_samples[0].tolist()
			if len(sel_samples_regular) == 0:
				continue

			### deal with flip image
			P_rpn = model_rpn.predict_on_batch(X_flip)

			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,max_boxes=300)

			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2_flip, Y1_flip, Y2_flip, Y_view_flip = roi_helpers.calc_iou_new(R, img_data_flip, C, class_mapping)


			pos_samples = np.where(Y_view_flip[0, :, -1] != 20)
			sel_samples_flip = pos_samples[0].tolist()
			if len(sel_samples_flip) == 0:
				continue


			loss_class = [0,0,0,0,0,0,0]
			X2,Y_view = roi_helpers.prep_roi_flip(X2[:, sel_samples_regular, :],Y_view[:, sel_samples_regular, :],C)
			X2_flip,Y_view_flip = roi_helpers.prep_roi_flip(X2_flip[:, sel_samples_flip, :],Y_view_flip[:, sel_samples_flip, :],C)
			### train classifier
			# loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples_regular, :]], [Y1[:, sel_samples_regular, :], Y2[:, sel_samples_regular, :],Y_view[:, sel_samples_regular, :]])

			#### train flip net
			# loss_flip = model_flip.train_on_batch([X, X2,X_flip, X2_flip,Y_view],[Y_view, Y_view_flip,np.array([[0]])])
			loss_flip = model_flip.train_on_batch([X, X2,X_flip, X2_flip,Y_view],[Y_view, Y_view_flip])
			if iter_num%500 == 0 :
				model_view_only.save_weights(weight_path_tmp)
				model_classifier.load_weights(weight_path_tmp,by_name=True)
				out_cls,out_reg,out_az = model_classifier.predict([X, X2])
				gt_label = np.argmax(Y1[:, sel_samples_regular, :],axis=2)
				gt_az = np.argmax(Y_view[0,:,:360],axis=1)
				az = []
				true_az = []
				cls = []
				true_cls =[]
				for ind in range(len(sel_samples_regular)):
					if gt_label[0,ind]!=20:
						az.append(np.argmax(out_az[0,ind,360*gt_label[0,ind]:360*(gt_label[0,ind]+1)],axis=0))
						true_az.append(gt_az[ind])
						cls.append(np.argmax(out_cls[0,ind,:],axis=0))
						true_cls.append(gt_label[0,ind])
				print('\n')
				print('true cls {} \n estimated cls {}'.format(true_cls,cls))
				print('true az {} \n estimated az {}'.format(true_az,az))
			# losses[iter_num, 0] = 0
			# losses[iter_num, 1] = 0
			# losses[iter_num, 0] = loss_rpn[1]
			# losses[iter_num, 1] = loss_rpn[2]
			# losses[iter_num, 2] = loss_class[1]
			# losses[iter_num, 3] = loss_class[2]
			# losses[iter_num, 4] = loss_class[3]
			# losses[iter_num, 5] = loss_class[4]
			# losses[iter_num, 4] = loss_class[0]
			# weight_t= np.mean(model_classifier.layers[39].get_weights()[0])
			iter_num += 1
			# progbar.update(iter_num, [('view_cls', loss_flip[1]),('view_flip', loss_flip[2]),('dist', loss_flip[3])])
			progbar.update(iter_num, [('view_cls', loss_flip[1]),('view_flip', loss_flip[2])])
			# progbar.update(iter_num, [('view_cls', losses[:iter_num, 4].sum(0)/(losses[:iter_num, 4]!=0).sum(0)),('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
			# 						  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])
			# progbar.update(iter_num, [('view_cls', losses[:iter_num, 4].sum(0)/(losses[:iter_num, 4]!=0).sum(0))])
				# print('iter time {}'.format(time.time()-t_start))

			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				loss_class_view = np.mean(losses[:, 4])
				class_acc = np.mean(losses[:, 5])


				mean_overlapping_bboxes = 0
				# mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				# rpn_accuracy_for_epoch = []

				if C.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Loss Detector view: {}'.format(loss_class_view))
					print('Elapsed time: {}'.format(time.time() - start_time))

				# curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				countNum = 0
				start_time = time.time()

				# if curr_loss < best_loss:
				# 	if C.verbose:
				# 		print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
				# 	save_epoch = epoch_num
				# 	best_loss = curr_loss
				# 	# model_view_only.save_weights(weight_path_tmp)
				# 	# model_classifier.load_weights(weight_path_tmp,by_name=True)
				# 	model_all.save_weights(C.model_path)

				# print('last save was at epoch {}'.format(save_epoch))
				print ('best accuracy {} in epoch {}'.format(best_succ,best_succ_epoch))
				print('MAP is {}'.format(MAP))
				## save weight every x epochs
				if epoch_num%epoch_save_num == 0 and epoch_num !=0:
					model_view_only.save_weights(weight_path_tmp)
					model_classifier.load_weights(weight_path_tmp, by_name=True)
					temp_ind = C.model_path.index(".hdf5")
					C.model_path_epoch = C.model_path[:temp_ind] + '_epoch_{}'.format(epoch_num) + C.model_path[temp_ind:]
					tmp_succ,MAP =test_view_func(C, model_rpn, model_classifier)
					succ_vec[0, int(epoch_num / epoch_save_num)] =tmp_succ
					# model_view_only.save_weights(weight_path_tmp)
					# model_classifier.load_weights(weight_path_tmp,by_name=True)
					if np.max(succ_vec) == succ_vec[0,int(epoch_num/epoch_save_num)]:
						best_succ = succ_vec[0,int(epoch_num/epoch_save_num)]
						best_succ_epoch = epoch_num
						model_all.save_weights(C.model_path_epoch)

				###test azimuth

				break

		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print('Training complete, exiting.')
