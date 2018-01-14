from __future__ import division
import os
### to work on cpu
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import sys
import pickle,cPickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.resnet_FC as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers, data_generators
from PIL import Image
import pylab
import imageio
from sklearn.neighbors.classification import KNeighborsClassifier




sys.setrecursionlimit(40000)

old_cls = ['aeroplane','bicycle','boat','bottle','bus','car','chair','diningtable','motorbike','sofa','train', 'tvmonitor']
class_mapping = {'sheep': 5, 'bottle': 8, 'horse': 15, 'bg': 20, 'bicycle': 17, 'motorbike': 16, 'cow': 12, 'sofa': 14, 'dog': 0, 'bus': 11, 'cat': 10, 'person': 6, 'train': 4, 'diningtable': 13, 'aeroplane': 19, 'car': 1, 'pottedplant': 2, 'tvmonitor': 7, 'chair': 9, 'bird': 3, 'boat': 18}

inv_class_mapping = {v: k for k, v in class_mapping.iteritems()}
parser = OptionParser()

# parser.add_option("-p", "--path", dest="test_path", help="Path to test data.",default ='/home/gilad/ssd/keras-frcnn-master/test_data')
parser.add_option("-p", "--path", dest="test_path", help="Path to test data.",default ='/home/gilad/bar')

parser.add_option("-n", "--num_rois", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata1 related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--im_path", dest="im_path", help="Path to test data txt file.",default ='/home/gilad/Dropbox/Project/CNN/viewNet/data_txt/bus_wGT_original.txt')
parser.add_option("--ep", dest="epoch_num", help="epoch number when calling this from train frcnn",default =1)
parser.add_option("--weight", dest="weight_name", help="weight name when calling this from train frcnn",default ='model_frcnn_new.hdf5')
parser.add_option("--input_train_file", dest="input_train_file", help="if there is a pickle file for train data.",default='keras_frcnn/train_data_Wflip_pascal.pickle' )

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

test_cls = 'aeroplane'

## read the training data from pickle file or from annotations
test_pickle = '/home/gilad/ssd/keras-frcnn-master/test_data_{}.pickle'.format(test_cls)
if os.path.exists(test_pickle):
	with open(test_pickle) as f:
		all_imgs, classes_count,_ = pickle.load(f)




backend = K.image_dim_ordering()
filename ='/home/gilad/bar/real7.p'
video_filename = "/home/gilad/ssd/keras-frcnn-master/a.mp4"
gt_cls_num = class_mapping[test_cls]
print( 'work on class {}'.format(test_cls))
base_path = os.getcwd()
# weight_path = '/home/gilad/ssd/keras-frcnn-master/model_frcnn_new.hdf5'
# weight_path = '/home/gilad/ssd/keras-frcnn-master/model_frcnn_siam_50_epoch_280.hdf5'
# weight_path =os.path.join(base_path,'model_trip_randselect_1by1_epoch_20.hdf5')
weight_path =os.path.join(base_path,'models/model_FC_NN.hdf5')
# weight_path = '/home/gilad/ssd/keras-frcnn-master/model_frcnn_siam_view_2.hdf5'
# weight_path = os.path.join(os.getcwd(),options.weight_name)

# turn off any data augmentation at test time
save_flag = False
visualise = False
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
count = 0
good_img = 0
not_good = 0
img_path = options.test_path

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def draw_bbox(img,bbox,prob,azimuth,ratio):
	# new_boxes, new_probs, new_az = roi_helpers.non_max_suppression_fast(bbox, prob, azimuth, overlap_thresh=0.3,use_az=True)
	new_boxes=bbox
	new_az = azimuth
	new_probs = prob
	for jk in range(new_boxes.shape[0]):
		(x1, y1, x2, y2) = new_boxes[jk, :]

		(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

		cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
					  (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)
		# cv2.rectangle(img,(bbox_gt['x1'], bbox_gt['y1']), (bbox_gt['x2'], bbox_gt['y2']), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

		# textLabel = '{}: {},azimuth : {}'.format(key,int(100*new_probs[jk]),new_az[jk])
		textLabel = 'azimuth : {}'.format(new_az[jk])

		all_dets.append((key, 100 * new_probs[jk]))

		(retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
		textOrg = (real_x1, real_y1 + 15)

		cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
					  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
		cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
					  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
		cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

def display_image(img):
	img1 = img[:, :, (2, 1, 0)]
	# img1=img
	im = Image.fromarray(img1.astype('uint8'), 'RGB')
	im.show()

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
## read the training data from pickle file or from annotations
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))
	return (real_x1, real_y1, real_x2 ,real_y2)


vnum_test = 24
azimuth_vec = np.concatenate(
    ([0], np.linspace((360. / (vnum_test * 2)), 360. - (360. / (vnum_test * 2)), vnum_test)),
    axis=0)

def find_interval(azimuth, azimuth_vec):
    for i in range(len(azimuth_vec)):
        if azimuth < azimuth_vec[i]:
            break
    ind = i
    if azimuth > azimuth_vec[-1]:
        ind = 1
    return ind

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)


if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (1024, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, 1024)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)


base_output = nn.nn_base(img_input, trainable=True)
model_base = Model(img_input,base_output)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(base_output, num_anchors)

classifier_only,inner = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=21,trainable_cls=False,trainable_view=False)
model_classifier_only = Model([feature_map_input, roi_input],classifier_only)



model_inner = Model([feature_map_input,roi_input],inner)
model_rpn = Model(img_input, rpn)
model_rpn_features = Model(img_input, rpn[2:])

model_classifier = Model([feature_map_input, roi_input],classifier_only)
# model_base.load_weights(weight_path, by_name=True)
model_classifier_only.load_weights(weight_path, by_name=True)
model_rpn.load_weights(weight_path, by_name=True)
model_classifier.load_weights(weight_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')
## read the training data from pickle file or from annotations
test_pickle = os.path.join(base_path,'pickle_data/test_data_{}.pickle'.format(test_cls))
if os.path.exists(test_pickle):
	with open(test_pickle) as f:
		all_imgs, classes_count, class_mapping = pickle.load(f)


obj_num = 0
bbox_threshold_orig = 0.6
th_bbox = 0.3
#### open images from folder


# for idx, img_name in enumerate(sorted(os.listdir(img_path))):
# 	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
# 		continue
# 	print(img_name)
# 	filepath = os.path.join(img_path,img_name)
# 	img = cv2.imread(filepath)caricycle

#### open images from file

## get GT for all az for single cls
# feature_az = []
# sorted_path = options.input_train_file
# tmp_ind = sorted_path.index('.pickle')
# sorted_path = sorted_path[:tmp_ind]+"_sorted_Angles"+sorted_path[tmp_ind:]
# if os.path.exists(sorted_path):
# 	print("loading sorted data")
# 	with open(sorted_path) as f:
# 		trip_data = pickle.load(f)
# im_file = []
# ind = []
# for ii in range(360):
# 	try:
# 		im_file.append(trip_data[test_cls][ii][0])
# 		ind.append(ii)
# 	except:
# 		print('no azimuth {}'.format(ii))
# data_gen_train = data_generators.get_anchor_gt(im_file, [], C, K.image_dim_ordering(), mode='test')
# azimuth_dict = []
# inner_NN = []
# for tt in ind:
# 	try:
# 		# print ('im num {}'.format(good_img))
# 		X, Y, img_data = next(data_gen_train)
#
# 		P_rpn = model_rpn.predict_on_batch(X)
#
# 		R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,
# 								   max_boxes=300)
#
# 		X2, Y1, Y2, Y_view = roi_helpers.calc_iou_new(R, img_data, C, C.class_mapping)
#
#
#
# 		pos_samples = np.where(Y1[0, :, -1] == 0)
# 		sel_samples = pos_samples[0].tolist()
# 		R = X2[0,sel_samples,:]
# 		for jk in range(R.shape[0] // C.num_rois + 1):
# 			ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
# 			if ROIs.shape[1] == 0:
# 				break
#
# 			if jk == R.shape[0] // C.num_rois:
# 				# pad R
# 				curr_shape = ROIs.shape
# 				target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
# 				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
# 				ROIs_padded[:, :curr_shape[1], :] = ROIs
# 				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
# 				ROIs = ROIs_padded
#
# 			[P_cls, P_regr, P_view] = model_classifier_only.predict([P_rpn[2], ROIs])
# 			iner_f = model_inner.predict([P_rpn[2], ROIs])
# 			# oo = model_classifier_only.predict([F, ROIs])
#
#
# 			for ii in range(R.shape[0]):
#
# 				if np.max(P_cls[0, ii, :]) < bbox_threshold_orig or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
# 					continue
#
# 				## get class from the net
# 				# cls_num = np.argmax(P_cls[0, ii, :])
#
# 				## use gt class
# 				cls_num = gt_cls_num
#
# 				cls_name = inv_class_mapping[cls_num]
# 				cls_view = P_view[0, ii, 360 * cls_num:360 * (cls_num + 1)]
#
#
# 				# azimuths[cls_name].append(np.argmax(cls_view, axis=0))
# 				inner_NN.append(iner_f[0,ii,:])
# 				azimuth_dict.append(img_data['bboxes'][0]['azimuth'])
# 	except:
# 		print('failed on az {}'.format(img_data['bboxes'][0]['azimuth']))
# ## calculating some mean feature map for every az
# with open('NN.pickle','w') as f:
# 	pickle.dump([inner_NN,azimuth_dict],f)
# 	print('saved PICKLE')

with open('NN.pickle') as f:
	inner_NN,azimuth_dict = pickle.load(f)
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(inner_NN, azimuth_dict)

jj = 0
for im_file in all_imgs:
	jj += 1
	if jj%50 == 0:
		print(jj)
	filepath = im_file['filepath']
	img = cv2.imread(filepath)
	img_gt = np.copy(img)
	if img is None:
		not_good += 1
		continue
	else:
		good_img += 1
		# print ('im num {}'.format(good_img))
	X, ratio = format_img(img, C)

	if backend == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
	# # convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	width,height = int(im_file["width"]), int(im_file["height"])
	resized_width, resized_height= data_generators.get_new_img_size(width, height, C.im_size)
	# [_,_, F] = model_rpn.predict(X)

	## pass on all the labels in the image, some of them are not equal to test_cls
	for bbox_gt in im_file['bboxes']:
		no_bbox_flag = 1
		bbox_threshold = bbox_threshold_orig
		if not bbox_gt['class']==test_cls:
			continue
		while no_bbox_flag and bbox_threshold > th_bbox:
			cls_gt = bbox_gt['class']
			az_gt = bbox_gt['azimuth']
			el_gt = bbox_gt['elevation']
			t_gt = bbox_gt['tilt']

			# apply the spatial pyramid pooling to the proposed regions
			bboxes = {}
			probs = {}
			azimuths ={}
			inner_res = {}
			if bbox_gt['class'] == test_cls and bbox_threshold == bbox_threshold_orig :
				obj_num += 1
				# print ('obj num {}'.format(obj_num))

			for jk in range(R.shape[0]//C.num_rois + 1):
				ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
				if ROIs.shape[1] == 0:
					break

				if jk == R.shape[0]//C.num_rois:
					#pad R
					curr_shape = ROIs.shape
					target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
					ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
					ROIs_padded[:, :curr_shape[1], :] = ROIs
					ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
					ROIs = ROIs_padded

				[P_cls, P_regr,P_view] = model_classifier_only.predict([F, ROIs])
				inner_out = model_inner.predict([F, ROIs])
				# oo = model_classifier_only.predict([F, ROIs])


				for ii in range(P_cls.shape[1]):

					if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
						continue

					## get class from the net
					# cls_num = np.argmax(P_cls[0, ii, :])

					## use gt class
					cls_num = gt_cls_num

					cls_name = inv_class_mapping[cls_num]
					cls_view = P_view[0, ii, 360*cls_num:360*(cls_num+1)]


					if cls_name not in bboxes:
						bboxes[cls_name] = []
						probs[cls_name] = []
						azimuths[cls_name] = []
						inner_res[cls_name] = []

					(x, y, w, h) = ROIs[0, ii, :]

					try:
						(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
						tx /= C.classifier_regr_std[0]
						ty /= C.classifier_regr_std[1]
						tw /= C.classifier_regr_std[2]
						th /= C.classifier_regr_std[3]
						x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
					except:
						pass
					bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
					probs[cls_name].append(np.max(P_cls[0, ii, :]))
					azimuths[cls_name].append(np.argmax(cls_view,axis=0))
					inner_res[cls_name].append(inner_out[0,ii,:])


			all_dets = []
			if len(bboxes)==0:
				bbox_threshold -= 0.1
			# cv2.rectangle(img_gt, (bbox_gt['x1'], bbox_gt['y1']), (bbox_gt['x2'], bbox_gt['y2']), (int(class_to_color[test_cls][0]), int(class_to_color[test_cls][1]), int(class_to_color[test_cls][2])), 2)
			for key in bboxes:
				# if 1:
				if key == test_cls and bbox_gt['class'] == test_cls:
					bbox = np.array(bboxes[key])
					prob = np.array(probs[key])
					azimuth = np.array(azimuths[key])
					inner_result = np.array(inner_res[key])
					# img = draw_bbox(img,bbox, prob, azimuth, ratio)
					azimuth = neigh.predict(inner_result)
					## get the azimuth from bbox that have more than 'overlap_thresh' overlap with gt_bbox
					az =[]
					overlap_thresh = 0.5
					try:
						while np.size(az)==0 and overlap_thresh>0:
							_,prob_bbox,az=roi_helpers.overlap_with_gt(bbox, prob, azimuth, bbox_gt,ratio=ratio, overlap_thresh=overlap_thresh, max_boxes=300, use_az=True)
							overlap_thresh-=0.1
						if overlap_thresh == 0:
							print("No good Bbox was found")
						counts = np.bincount(az)
					except:
						az=[];counts =[]
					try:
						az_fin = np.argmax(counts)
						true_bin = find_interval(az_gt, azimuth_vec)
						prob_bin = find_interval(az_fin, azimuth_vec)
						no_bbox_flag = 0
						if true_bin == prob_bin:
							count += 1
							break
					except:
						print('here')
						no_bbox_flag = 1
						bbox_threshold -= 0.1

				## azimuth calculations



				## display

				# print('Elapsed time = {}'.format(time.time() - st))
				# print('real az is {}'.format(az_gt))
				# print('pred az is {}'.format(new_az[:]))
				# print(all_dets)
				if visualise:
					display_image(img)
				# cv2.imshow('img', img)
				# cv2.waitKey(0)
				if save_flag:
				   cv2.imwrite('./results_imgs/{}'.format(img_name),img)
				   # img = img[:, :, (2, 1, 0)]
				   # cv2.imwrite('./results_imgs/video/{}.png'.format(num),img)
				# print('save')
			bbox_threshold -= 0.1
			if visualise:
				display_image(img)
succ = float(count)/float(obj_num)*100.
print('for class {} -true count is {} out of {} from {} images . {} success'.format(test_cls,count,obj_num,good_img,succ))
f = open('test_val.txt','a')
f.write('epoch {} secc {}'.format(options.epoch_num,succ))
f.close()