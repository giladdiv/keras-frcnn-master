import numpy as np
import pdb
import math
from . import data_generators
import copy


band_width = 6
sigma= 2
vector_len =360
weight_norm = 0
for i in range(-band_width / 2, band_width / 2 + 1):
	weight_norm += np.exp(-np.abs(i) / sigma)
weight_flag = True

print ('******* Weight flag is {} ******'.format(weight_flag))
def weight_vector(ind,band_width =20,sigma=2,vector_len =360,batch_size=1,weight_norm =1):
    w_vec = np.zeros([batch_size,vector_len])
    for k in range(batch_size):
        for j in range(ind-band_width/2,ind+band_width/2+1):
			i = j%vector_len

			## gaussian
			w_vec[0,i] = np.exp(-np.abs(ind-j)/sigma)/weight_norm
			## not gaussian
			# w_vec[0,i] = 1 - 0.1*abs(ind-j) ## decrease in the BW with 0.1 factor
	return (w_vec)

def prep_flip(R,Y,C):
	'''
	fill ROI with R iterativly
	:param R:
	:param C:
	:return:
	'''
	if R.ndim == 2:
		ROIs = np.expand_dims(R, axis=0)
	else:
		ROIs = R

	if R.shape[0]!= C.num_rois:
		# pad R with the first value of R
		curr_shape = ROIs.shape
		target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
		ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
		Y_padded = np.zeros([curr_shape[0], C.num_rois, Y.shape[-1]]).astype(Y.dtype)
		ROIs_padded[:, :curr_shape[1], :] = ROIs
		Y_padded[:, :Y.shape[1], :] = Y

		iter_num,rem = divmod(C.num_rois,curr_shape[1])
		for ii in range(iter_num):
			ROIs_padded[0, curr_shape[1]*ii:curr_shape[1]*(ii+1), :] = ROIs[0, :, :]
			Y_padded[0, curr_shape[1]*ii:curr_shape[1]*(ii+1), :] = Y[0,:,:]
		if rem > 0:
			ROIs_padded[0, curr_shape[1]*iter_num:, :] = ROIs[0, :rem, :]
			Y_padded[0, curr_shape[1]*iter_num:, :] = Y[0, :rem, :]
		ROIs = ROIs_padded
		Y = Y_padded
	return ROIs,Y




def prep_roi_siam(R,C):
	'''
	fill ROI with R iterativly
	:param R:
	:param C:
	:return:
	'''
	if R.ndim == 2:
		ROIs = np.expand_dims(R, axis=0)
	else:
		ROIs = R

	if R.shape[0]!= C.num_rois:
		# pad R with the first value of R
		curr_shape = ROIs.shape
		target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
		ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
		ROIs_padded[:, :curr_shape[1], :] = ROIs
		iter_num,rem = divmod(C.num_rois,curr_shape[1])
		for ii in range(iter_num):
			ROIs_padded[0, curr_shape[1]*ii:curr_shape[1]*(ii+1), :] = ROIs[0, :, :]
		if rem>0:
			ROIs_padded[0, curr_shape[1]*iter_num:, :] = ROIs[0, :rem, :]
		ROIs = ROIs_padded
	return ROIs

def calc_iou(R, img_data, C, class_mapping):

	bboxes = img_data['bboxes']
	(width, height) = (img_data['width'], img_data['height'])
	# get image dimensions for resizing
	(resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

	gta = np.zeros((len(bboxes), 4))

	for bbox_num, bbox in enumerate(bboxes):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
		gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

	x_roi = []
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []

	for ix in range(R.shape[0]):
		(x1, y1, x2, y2) = R[ix, :]
		x1 = int(round(x1))
		y1 = int(round(y1))
		x2 = int(round(x2))
		y2 = int(round(y2))

		best_iou = 0.0
		best_bbox = -1
		for bbox_num in range(len(bboxes)):
			curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
			if curr_iou > best_iou:
				best_iou = curr_iou
				best_bbox = bbox_num

		if best_iou < C.classifier_min_overlap:
				continue
		else:
			w = x2 - x1
			h = y2 - y1
			x_roi.append([x1, y1, w, h])

			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
				# hard negative example
				cls_name = 'bg'
			elif C.classifier_max_overlap <= best_iou:
				cls_name = bboxes[best_bbox]['class']
				cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
				cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

				cx = x1 + w / 2.0
				cy = y1 + h / 2.0

				tx = (cxg - cx) / float(w)
				ty = (cyg - cy) / float(h)
				tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
				th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError

		class_num = class_mapping[cls_name]
		class_label = len(class_mapping) * [0]
		class_label[class_num] = 1
		y_class_num.append(copy.deepcopy(class_label))
		coords = [0] * 4 * (len(class_mapping) - 1)
		labels = [0] * 4 * (len(class_mapping) - 1)
		if cls_name != 'bg':
			label_pos = 4 * class_num
			sx, sy, sw, sh = C.classifier_regr_std
			coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
			labels[label_pos:4+label_pos] = [1, 1, 1, 1]
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))
		else:
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))

	if len(x_roi) == 0:
		return None, None, None

	X = np.array(x_roi)
	Y1 = np.array(y_class_num)
	Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

	return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0)

def calc_iou_new(R, img_data, C, class_mapping):

	bboxes = img_data['bboxes']
	(width, height) = (img_data['width'], img_data['height'])

	# get image dimensions for resizing
	(resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

	gta = np.zeros((len(bboxes), 4))

	for bbox_num, bbox in enumerate(bboxes):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
		gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

	x_roi = []
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []
	y_view =[]

	for ix in range(R.shape[0]):
		(x1, y1, x2, y2) = R[ix, :]
		x1 = int(round(x1))
		y1 = int(round(y1))
		x2 = int(round(x2))
		y2 = int(round(y2))

		best_iou = 0.0
		best_bbox = -1
		for bbox_num in range(len(bboxes)):
			curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
			if curr_iou > best_iou:
				best_iou = curr_iou
				best_bbox = bbox_num

		if best_iou < C.classifier_min_overlap:
				continue
		else:
			w = x2 - x1
			h = y2 - y1
			x_roi.append([x1, y1, w, h])

			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
				# hard negative example
				cls_name = 'bg'
			elif C.classifier_max_overlap <= best_iou:
				cls_name = bboxes[best_bbox]['class']
				cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
				cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

				cx = x1 + w / 2.0
				cy = y1 + h / 2.0

				tx = (cxg - cx) / float(w)
				ty = (cyg - cy) / float(h)
				tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
				th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError
		numOfCls = len(class_mapping)
		class_num = class_mapping[cls_name]
		class_label =  numOfCls* [0]
		class_label[class_num] = 1
		# y_class_num.append(copy.deepcopy(class_label)) ### was in use till 28.8
		y_class_num.append(class_label)



		## view point label (num_roi,[360-for view information,360*num_cls-maskcls to take the right part of the vector],cls)
		if not(bboxes[best_bbox]['viewpoint_data']) or cls_name == 'bg' or cls_name == 'bottle':
			class_mask = np.kron(class_label,np.ones([1,360]))
			class_mask.astype(np.int)
			class_mask = class_mask.tolist()
			view_label_temp = 360 *[0.]
			view_label = view_label_temp +class_mask[0]+[numOfCls-1]
		else:
			if weight_flag:
				class_mask = np.kron(class_label, np.ones([1, 360]))
				class_mask.astype(np.int)
				class_mask = class_mask.tolist()
				view_label_temp = weight_vector(ind =bboxes[best_bbox]['azimuth'],band_width =band_width,sigma=sigma,vector_len =360,batch_size=1,weight_norm =weight_norm)
				view_label = view_label_temp.tolist()[0] + class_mask[0] + [class_num]
			else:
				class_mask = np.kron(class_label,np.ones([1,360]))
				class_mask.astype(np.int)
				class_mask = class_mask.tolist()
				view_label_temp = 360 *[0.]
				view_label_temp[bboxes[best_bbox]['azimuth']] = 1. #place one in the 360 range that fits his class
				view_label = view_label_temp +class_mask[0]+[class_num]

		## y_view = [view,mask]
		y_view.append(view_label)
		##

		coords = [0] * 4 * (len(class_mapping) - 1)
		labels = [0] * 4 * (len(class_mapping) - 1)
		if cls_name != 'bg':
			label_pos = 4 * class_num
			sx, sy, sw, sh = C.classifier_regr_std
			coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
			labels[label_pos:4+label_pos] = [1, 1, 1, 1]
			# y_class_regr_coords.append(copy.deepcopy(coords))  ### was in use till 28.8
			# y_class_regr_label.append(copy.deepcopy(labels)) ### was in use till 28.8
			y_class_regr_coords.append(coords)
			y_class_regr_label.append(labels)
		else:
			# y_class_regr_coords.append(copy.deepcopy(coords)) ### was in use till 28.8
			# y_class_regr_label.append(copy.deepcopy(labels) )### was in use till 28.8
			y_class_regr_coords.append(coords)
			y_class_regr_label.append(labels)

	if len(x_roi) == 0:
		return None, None, None

	X = np.array(x_roi)
	Y1 = np.array(y_class_num)
	Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)
	Y_view = np.array(y_view)

	return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0),np.expand_dims(Y_view, axis=0)

def calc_iou_siam(R, img_data, C, class_mapping):

	bboxes = img_data['bboxes']
	(width, height) = (img_data['width'], img_data['height'])

	# get image dimensions for resizing
	(resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)
	x_roi = []
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []
	y_view =[]

	for ix in range(R.shape[0]):
		(x1, y1, x2, y2) = R[ix, :]
		x1 = int(round(x1))
		y1 = int(round(y1))
		x2 = int(round(x2))
		y2 = int(round(y2))

		best_iou = 0.0
		best_bbox = -1
		for bbox_num in range(len(bboxes)):
			curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
			if curr_iou > best_iou:
				best_iou = curr_iou
				best_bbox = bbox_num

		if best_iou < C.classifier_min_overlap:
				continue
		else:
			w = x2 - x1
			h = y2 - y1
			x_roi.append([x1, y1, w, h])

			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
				# hard negative example
				cls_name = 'bg'
			elif C.classifier_max_overlap <= best_iou:
				cls_name = bboxes[best_bbox]['class']
				cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
				cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

				cx = x1 + w / 2.0
				cy = y1 + h / 2.0

				tx = (cxg - cx) / float(w)
				ty = (cyg - cy) / float(h)
				tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
				th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError
		numOfCls = len(class_mapping)
		class_num = class_mapping[cls_name]
		class_label =  numOfCls* [0]
		class_label[class_num] = 1
		# y_class_num.append(copy.deepcopy(class_label)) ### was in use till 28.8
		y_class_num.append(class_label)



		## view point label (num_roi,[360-for view information,360*num_cls-maskcls to take the right part of the vector],cls)
		if not(bboxes[best_bbox]['viewpoint_data']) or cls_name == 'bg' or cls_name == 'bottle':
			class_mask = np.kron(class_label,np.ones([1,360]))
			class_mask.astype(np.int)
			class_mask = class_mask.tolist()
			view_label_temp = 360 *[0.]
			view_label = view_label_temp +class_mask[0]+[numOfCls-1]
		else:
			if weight_flag:
				class_mask = np.kron(class_label, np.ones([1, 360]))
				class_mask.astype(np.int)
				class_mask = class_mask.tolist()
				view_label_temp = weight_vector(ind =bboxes[best_bbox]['azimuth'],band_width =band_width,sigma=sigma,vector_len =360,batch_size=1,weight_norm =weight_norm)
				view_label = view_label_temp.tolist()[0] + class_mask[0] + [class_num]
			else:
				class_mask = np.kron(class_label,np.ones([1,360]))
				class_mask.astype(np.int)
				class_mask = class_mask.tolist()
				view_label_temp = 360 *[0.]
				view_label_temp[bboxes[best_bbox]['azimuth']] = 1. #place one in the 360 range that fits his class
				view_label = view_label_temp +class_mask[0]+[class_num]

		## y_view = [view,mask]
		y_view.append(view_label)
		##

		coords = [0] * 4 * (len(class_mapping) - 1)
		labels = [0] * 4 * (len(class_mapping) - 1)
		if cls_name != 'bg':
			label_pos = 4 * class_num
			sx, sy, sw, sh = C.classifier_regr_std
			coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
			labels[label_pos:4+label_pos] = [1, 1, 1, 1]
			# y_class_regr_coords.append(copy.deepcopy(coords))  ### was in use till 28.8
			# y_class_regr_label.append(copy.deepcopy(labels)) ### was in use till 28.8
			y_class_regr_coords.append(coords)
			y_class_regr_label.append(labels)
		else:
			# y_class_regr_coords.append(copy.deepcopy(coords)) ### was in use till 28.8
			# y_class_regr_label.append(copy.deepcopy(labels) )### was in use till 28.8
			y_class_regr_coords.append(coords)
			y_class_regr_label.append(labels)

	if len(x_roi) == 0:
		return None, None, None

	X = np.array(x_roi)
	Y1 = np.array(y_class_num)
	Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)
	Y_view = np.array(y_view)

	return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0),np.expand_dims(Y_view, axis=0)


def apply_regr(x, y, w, h, tx, ty, tw, th):
	try:
		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		w1 = math.exp(tw) * w
		h1 = math.exp(th) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.
		x1 = int(round(x1))
		y1 = int(round(y1))
		w1 = int(round(w1))
		h1 = int(round(h1))

		return x1, y1, w1, h1

	except ValueError:
		return x, y, w, h
	except OverflowError:
		return x, y, w, h
	except Exception as e:
		print(e)
		return x, y, w, h

def apply_regr_np(X, T):
	try:
		x = X[0, :, :]
		y = X[1, :, :]
		w = X[2, :, :]
		h = X[3, :, :]

		tx = T[0, :, :]
		ty = T[1, :, :]
		tw = T[2, :, :]
		th = T[3, :, :]

		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy

		w1 = np.exp(tw) * w
		h1 = np.exp(th) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.

		x1 = np.round(x1)
		y1 = np.round(y1)
		w1 = np.round(w1)
		h1 = np.round(h1)
		return np.stack([x1, y1, w1, h1])
	except Exception as e:
		print(e)
		return X

def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)


def overlap_mAVP(boxes,pred_bbox):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	x1_pred,y1_pred,x2_pred,y2_pred = pred_bbox['x1'],pred_bbox['y1'],pred_bbox['x2'],pred_bbox['y2']

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")


	# calculate the areas
	area = (x2 - x1) * (y2 - y1)
	area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)

	xx1_int = np.maximum(x1_pred, x1)
	yy1_int = np.maximum(y1_pred, y1)
	xx2_int = np.minimum(x2_pred, x2)
	yy2_int = np.minimum(y2_pred, y2)

	ww_int = np.maximum(0, xx2_int - xx1_int)
	hh_int = np.maximum(0, yy2_int - yy1_int)

	area_int = ww_int * hh_int

	# find the union
	area_union = area_pred + area - area_int

	# compute the ratio of overlap
	overlap = area_int/(area_union + 1e-6)

	return overlap





def overlap_with_gt(boxes, probs,azimuth,gt_bbox,ratio=1, overlap_thresh=0.9, max_boxes=300,use_az = False,return_overlap = False):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	x1_gt,y1_gt,x2_gt,y2_gt = get_real_coordinates(1./ratio,gt_bbox['x1'],gt_bbox['y1'],gt_bbox['x2'],gt_bbox['y2'])

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# calculate the areas
	area = (x2 - x1) * (y2 - y1)
	area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)
	# sort the bounding boxes
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list

	# grab the last index in the indexes list and add the
	# index value to the list of picked indexes
	# last = len(idxs) - 1
	# ## in case there is only one good bbox
	# if last == 0:
	# 	last =1
		# print("only one Bbox")
	# i = idxs[last]
	# pick.append(i)

	# find the intersection

	xx1_int = np.maximum(x1_gt, x1[idxs])
	yy1_int = np.maximum(y1_gt, y1[idxs])
	xx2_int = np.minimum(x2_gt, x2[idxs])
	yy2_int = np.minimum(y2_gt, y2[idxs])

	ww_int = np.maximum(0, xx2_int - xx1_int)
	hh_int = np.maximum(0, yy2_int - yy1_int)

	area_int = ww_int * hh_int

	# find the union
	area_union = area_gt + area[idxs] - area_int

	# compute the ratio of overlap
	overlap = area_int/(area_union + 1e-6)

	good_idx = np.where(overlap > overlap_thresh)[0]
	pick = idxs[good_idx]

	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	overlap = overlap[pick]
	if use_az:
		azimuth = azimuth[pick]
		if return_overlap:
			return boxes, probs, azimuth,overlap
		else:
			return boxes, probs, azimuth
	else:
		return boxes, probs


def non_max_suppression_fast(boxes, probs,azimuth=0, overlap_thresh=0.9, max_boxes=300,use_az = False):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# calculate the areas
	area = (x2 - x1) * (y2 - y1)

	# sort the bounding boxes 
	idxs = np.argsort(probs)
	idx_az = []
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)

		area_int = ww_int * hh_int

		# find the union
		area_union = area[i] + area[idxs[:last]] - area_int

		# compute the ratio of overlap
		overlap = area_int/(area_union + 1e-6)

		# delete all indexes from the index list that have
		tmp_idx = np.concatenate(([last],np.where(overlap > overlap_thresh)[0]))
		idx_az.append(idxs[tmp_idx])
		idxs = np.delete(idxs,tmp_idx )

		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	if use_az:
		az_count = []
		for az_id in idx_az:
			az_count.append(np.argmax(np.bincount(azimuth[az_id])))
		return boxes, probs, az_count
	else:
		return boxes, probs

import time
def rpn_to_roi(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9):

	regr_layer = regr_layer / C.std_scaling

	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios

	assert rpn_layer.shape[0] == 1

	if dim_ordering == 'th':
		(rows,cols) = rpn_layer.shape[2:]

	elif dim_ordering == 'tf':
		(rows, cols) = rpn_layer.shape[1:3]

	curr_layer = 0
	if dim_ordering == 'tf':
		A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
	elif dim_ordering == 'th':
		A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

	for anchor_size in anchor_sizes:
		for anchor_ratio in anchor_ratios:

			anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
			anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride
			if dim_ordering == 'th':
				regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
			else:
				regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
				regr = np.transpose(regr, (2, 0, 1))

			X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

			A[0, :, :, curr_layer] = X - anchor_x/2
			A[1, :, :, curr_layer] = Y - anchor_y/2
			A[2, :, :, curr_layer] = anchor_x
			A[3, :, :, curr_layer] = anchor_y

			if use_regr:
				A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

			A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
			A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
			A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

			A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
			A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
			A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

			curr_layer += 1

	all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
	all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

	x1 = all_boxes[:, 0]
	y1 = all_boxes[:, 1]
	x2 = all_boxes[:, 2]
	y2 = all_boxes[:, 3]

	idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

	all_boxes = np.delete(all_boxes, idxs, 0)
	all_probs = np.delete(all_probs, idxs, 0)

	result = non_max_suppression_fast(all_boxes, all_probs,azimuth=0, overlap_thresh=overlap_thresh, max_boxes=max_boxes,use_az =False)[0]

	return result

def az2vec(az,class_num,roi_num,class_mapping):
	# taking az and trasforming it to y, y.shape = [1, roi_num, [360+360*number of classes + 1] ]
	numOfCls = len(class_mapping)
	class_label = numOfCls * [0]
	class_label[class_num] = 1
	class_mask = np.kron(class_label, np.ones([1, 360]))
	class_mask.astype(np.int)
	class_mask = class_mask.tolist()
	view_label_temp = 360 * [0.]
	view_label_temp[az] = 1.  # place one in the 360 range that fits his class
	view_label = view_label_temp + class_mask[0] + [class_num]
	Y_view = np.expand_dims(np.tile(view_label,(roi_num,1)),axis=0)
	return Y_view

def compere_weights(w_a,w_b,start_a,start_b):
	vec_size =min(len(w_a)-start_a,len(w_b)-start_b)
	diff_vec = np.zeros([1,vec_size])
	for ii in range(vec_size):
		if (w_a[start_a+ii].shape == w_b[start_b+ii].shape):
			diff_vec[0,ii] = sum(w_a[start_a+ii].reshape([-1])-w_b[start_b+ii].reshape([-1]))
		else:
			diff_vec[0,ii] = 99
			break
	return diff_vec


def overlap_messa(bbox,gt_bbox,ratio=1):

	# grab the coordinates of the bounding boxes
	x1,y1,x2,y2 = (float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3]))
	x1_gt,y1_gt,x2_gt,y2_gt = (gt_bbox['x1'],gt_bbox['y1'],gt_bbox['x2'],gt_bbox['y2'])

	# np.testing.assert_array_less(x1, x2)
	# np.testing.assert_array_less(y1, y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	# if bbox.dtype.kind == "i":
	# 	bbox = bbox.astype("float")

	# initialize the list of picked indexes
	# calculate the areas
	area = (x2 - x1) * (y2 - y1)
	area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)
	# sort the bounding boxes


	xx1_int = np.maximum(x1_gt, x1)
	yy1_int = np.maximum(y1_gt, y1)
	xx2_int = np.minimum(x2_gt, x2)
	yy2_int = np.minimum(y2_gt, y2)

	ww_int = np.maximum(0, xx2_int - xx1_int)
	hh_int = np.maximum(0, yy2_int - yy1_int)

	area_int = ww_int * hh_int

	# find the union
	area_union = area_gt + area - area_int

	# compute the ratio of overlap
	overlap = area_int/(area_union + 1e-6)

	good_idx = overlap >= 0.5


	# return only the bounding boxes that were picked using the integer data type
	return good_idx

def overlap_ratio(boxes, probs, overlap_thresh=0.7):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# calculate the areas
	area = (x2 - x1) * (y2 - y1)

	# sort the bounding boxes
	idxs = np.argsort(probs)
	idx_overlap =[]
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)

		area_int = ww_int * hh_int

		# find the union
		area_union = area[i] + area[idxs[:last]] - area_int

		# compute the ratio of overlap
		overlap = area_int/(area_union + 1e-6)

		# delete all indexes from the index list that have
		idx_overlap.append(np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))
		idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))


	# return only the bounding boxes that were picked using the integer data type

	return idx_overlap