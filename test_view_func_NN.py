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


def test_view_func_NN(model_classifier,model_rpn,model_inner,C):
	test_cls = 'aeroplane'
	input_train_file = 'pickle_data/train_data_Wflip_all.pickle'

	## read the training data from pickle file or from annotations
	test_pickle = 'pickle_data/test_data_{}.pickle'.format(test_cls)
	if os.path.exists(test_pickle):
		with open(test_pickle) as f:
			all_imgs, classes_count,_ = pickle.load(f)

	class_mapping = C.class_mapping
	inv_class_mapping = {v: k for k, v in class_mapping.iteritems()}
	backend = K.image_dim_ordering()
	gt_cls_num = class_mapping[test_cls]
	print( 'work on class {}'.format(test_cls))
	base_path = os.getcwd()

	# turn off any data augmentation at test time
	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False
	count = 0
	good_img = 0
	not_good = 0

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
	# print(class_mapping)
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
	C.num_rois = 32


	obj_num = 0
	bbox_threshold_orig = 0.6
	th_bbox = 0.4

	## get GT for all az for single cls
	feature_az = []
	sorted_path = input_train_file
	tmp_ind = sorted_path.index('.pickle')
	sorted_path = sorted_path[:tmp_ind]+"_sorted_Angles"+sorted_path[tmp_ind:]
	if os.path.exists(sorted_path):
		print("loading sorted data")
		with open(sorted_path) as f:
			trip_data = pickle.load(f)
	im_file = []
	ind = []
	for ii in range(360):
		for jj in range(3):
			try:
				im_file.append(trip_data[test_cls][ii][jj])
				ind.append(ii)
			except:
				if jj == 0:
					print('no azimuth {}'.format(ii))
	data_gen_train = data_generators.get_anchor_gt(im_file, [], C, K.image_dim_ordering(), mode='test')
	azimuth_dict = []
	inner_NN = []
	azimuths =[]
	for tt in range(len(ind)):
		try:
			if tt%100 == 0:
				print ('worked on {}/{}'.format(tt,len(ind)))
			# print ('im num {}'.format(good_img))
			X, Y, img_data = next(data_gen_train)

			P_rpn = model_rpn.predict_on_batch(X)

			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,
									   max_boxes=300)

			X2, Y1, Y2, Y_view = roi_helpers.calc_iou_new(R, img_data, C, C.class_mapping)



			pos_samples = np.where(Y1[0, :, -1] == 0)
			sel_samples = pos_samples[0].tolist()
			R = X2[0,sel_samples,:]
			for jk in range(R.shape[0] // C.num_rois + 1):
				ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
				if ROIs.shape[1] == 0:
					break

				if jk == R.shape[0] // C.num_rois:
					# pad R
					curr_shape = ROIs.shape
					target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
					ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
					ROIs_padded[:, :curr_shape[1], :] = ROIs
					ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
					ROIs = ROIs_padded

				[P_cls, P_regr, P_view] = model_classifier.predict([X, ROIs])
				iner_f = model_inner.predict([X, ROIs])
				# oo = model_classifier_only.predict([F, ROIs])


				for ii in range(len(sel_samples)):

					if np.max(P_cls[0, ii, :]) < bbox_threshold_orig or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
						continue

					## get class from the net
					# cls_num = np.argmax(P_cls[0, ii, :])

					## use gt class
					cls_num = gt_cls_num

					cls_name = inv_class_mapping[cls_num]
					cls_view = P_view[0, ii, 360 * cls_num:360 * (cls_num + 1)]


					# azimuths[cls_name].append(np.argmax(cls_view, axis=0))
					inner_NN.append(iner_f[0,ii,:])
					azimuth_dict.append(img_data['bboxes'][0]['azimuth'])
		except:
			print('failed on az {}'.format(img_data['bboxes'][0]['azimuth']))
	## calculating some mean feature map for every az
	with open('pickle_data/{}_NN.pickle'.format(C.weight_name ),'w') as f:
		pickle.dump([inner_NN,azimuth_dict],f)
		print('saved PICKLE')

	with open('pickle_data/{}_NN.pickle'.format(C.weight_name )) as f:
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
		Y1, Y2 = model_rpn.predict(X)
		R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
		# # convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		width,height = int(im_file["width"]), int(im_file["height"])
		resized_width, resized_height= data_generators.get_new_img_size(width, height, C.im_size)
		# [_,_, F] = model_rpn.predict(X)
		ROIs =[]
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
				if len(ROIs)==0:
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

						[P_cls, P_regr,P_view] = model_classifier.predict([X, ROIs])
						inner_out = model_inner.predict([X, ROIs])
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
							# print('here')
							no_bbox_flag = 1
							bbox_threshold -= 0.1

					## azimuth calculations



					## display

				bbox_threshold -= 0.1

	succ = float(count)/float(obj_num)*100.
	print('for class {} -true count is {} out of {} from {} images . {} success'.format(test_cls,count,obj_num,good_img,succ))
	return succ