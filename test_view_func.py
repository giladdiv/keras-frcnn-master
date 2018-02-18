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
from keras import backend as K
from keras_frcnn import roi_helpers, data_generators
from PIL import Image
import pylab
import imageio



def test_view_func(C,model_rpn,model_classifier):
	base_dir = os.getcwd()
	test_cls_all = ['aeroplane','bus','motorbike']
	class_mapping = C.class_mapping
	inv_class_mapping = {v: k for k, v in class_mapping.iteritems()}
	backend = K.image_dim_ordering()
	filename ='/home/gilad/bar/real7.p'
	video_filename = "/home/gilad/ssd/keras-frcnn-master/a.mp4"
	write_flag = False
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

	# turn off any data augmentation at test time
	save_flag = False
	visualise = False
	count = 0
	good_img = 0
	not_good = 0
	mAP = 0
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


	# print(rep)
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
	## read the training data from pickle file or from annotations
	# class_mapping = C.class_mapping
	succ = []
	for test_cls in test_cls_all:
		good_img = 0
		not_good = 0
		count = 0
		obj_num = 0
		gt_cls_num = class_mapping[test_cls]
		print('work on class {}'.format(test_cls))
		test_pickle = os.path.join(base_dir,'pickle_data/test_data_{}.pickle'.format(test_cls))
		if os.path.exists(test_pickle):
			with open(test_pickle) as f:
				all_imgs, classes_count, _ = pickle.load(f)
		for im_file in all_imgs:
			filepath = im_file['filepath']
			img = cv2.imread(filepath)
			img_gt = np.copy(img)
			if img is None:
				not_good += 1
				continue
			else:
				good_img += 1
				# print ('im num {}'.format(good_img))
			if good_img % 50 ==0:
				print("worked on {} images".format(good_img))
			X, ratio = format_img(img, C)

			if backend == 'tf':
				X = np.transpose(X, (0, 2, 3, 1))

			# get the feature maps and output from the RPN
			[Y1, Y2] = model_rpn.predict(X)
			R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
			# # convert from (x1,y1,x2,y2) to (x,y,w,h)
			R[:, 2] -= R[:, 0]
			R[:, 3] -= R[:, 1]

			width,height = int(im_file["width"]), int(im_file["height"])
			resized_width, resized_height= data_generators.get_new_img_size(width, height, C.im_size)
			# [_,_, F] = model_rpn.predict(X)

			ROIs = []
			## pass on all the labels in the image, some of them are not equal to test_cls
			for bbox_gt in im_file['bboxes']:
				if not bbox_gt['class']==test_cls:
					continue
				no_bbox_flag = 1
				bbox_threshold = bbox_threshold_orig

				while no_bbox_flag and bbox_threshold > th_bbox:
					cls_gt = bbox_gt['class']
					az_gt = bbox_gt['azimuth']
					el_gt = bbox_gt['elevation']
					t_gt = bbox_gt['tilt']
					if bbox_gt['class'] == test_cls and bbox_threshold == bbox_threshold_orig:
						obj_num += 1
					if len(ROIs)==0:
						# apply the spatial pyramid pooling to the proposed regions
						bboxes = {}
						probs = {}
						azimuths ={}


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

							for ii in range(P_cls.shape[1]):

								if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
									continue

								## get class from the net
								# cls_num = np.argmax(P_cls[0, ii, :])

								## use gt class
								cls_num = gt_cls_num

								cls_name = inv_class_mapping[cls_num]
								cls_view = P_view[0, ii, 360*cls_num:360*(cls_num+1)]
								# cls_name_gt = cls_nimg = draw_bbox(img,bbox, prob, azimuth, ratio)ame
								# if cls_name == cls_name_gt:
								# 	print(np.argmax(cls_view,axis=0))
								if cls_name not in bboxes:
									bboxes[cls_name] = []
									probs[cls_name] = []
									azimuths[cls_name] = []

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


					all_dets = []
					if len(bboxes)==0:
						bbox_threshold -= 0.1
					for key in bboxes:
						# if 1:
						if key == test_cls and bbox_gt['class'] == test_cls:
							bbox = np.array(bboxes[key])
							prob = np.array(probs[key])
							azimuth = np.array(azimuths[key])

							## get the azimuth from bbox that have more than 'overlap_thresh' overlap with gt_bbox
							az =[]
							overlap_thresh = 0.5
							try:
								while np.size(az)==0 and overlap_thresh>0.3:
									_,prob_bbox,az=roi_helpers.overlap_with_gt(bbox, prob, azimuth, bbox_gt,ratio=ratio, overlap_thresh=overlap_thresh, max_boxes=300, use_az=True)
									if np.size(az) != 0 and overlap_thresh == 0.5:
										mAP += 1
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
						# if visualise:
						# 	display_image(img)
						# # cv2.imshow('img', img)
						# # cv2.waitKey(0)
						# if save_flag:
						#    cv2.imwrite('./results_imgs/{}'.format(img_name),img)
						#    # img = img[:, :, (2, 1, 0)]
						#    # cv2.imwrite('./results_imgs/video/{}.png'.format(num),img)
						# # print('save')
					bbox_threshold -= 0.1
					# if visualise:
					# 	display_image(img)
		succ.append(float(count)/float(obj_num)*100.)
		string = 'for class {} -true count is {} out of {} from {} images . {} success'.format(test_cls,count,obj_num,good_img,float(count)/float(obj_num)*100.)
		print(string)
		mAP = float(mAP)/float(obj_num)*100.
		print ("MAP is {}".format(mAP))
		# if write_flag:
		# 	f = open('{}_results.txt'.format(weight_name),'a')
		# 	f.write(string+'\n')
		# 	f.close()
	return succ,mAP