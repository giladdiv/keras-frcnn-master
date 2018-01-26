import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.resnet_FC as nn
from keras import backend as K
from keras.layers import Input,Lambda
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
import scipy.io as sio
from sklearn.metrics import average_precision_score
import copy
from sklearn.neighbors.classification import KNeighborsClassifier
import tensorflow as tf



def discretize(ang, nPoses):
	divAng = 360./nPoses
	discrete_ang = np.mod(np.ceil((ang-divAng/2.)/divAng),nPoses)+1;
	return discrete_ang

def get_mAVP(pred, gt, f,key= 'aeroplane'):
	T_view = {}
	T_bbox = {}
	P = {}
	fx, fy = f

	gt_new = []
	gt_bbox =[]
	ind = []
	for ii,gt_box in enumerate(gt):
		if not(gt_box['class'] != key or gt_box['difficult']):
			gt_box['bbox_matched'] = False
			gt_box['view_matched'] = False
			gt_bbox.append([float(gt_box['x1']),float(gt_box['y1']),float(gt_box['x2']),float(gt_box['y2'])])
			gt_new.append(gt_box)


	pred_probs = np.array([s['prob'] for s in pred])
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:

		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		pred_prob = pred_box['prob']
		if pred_class not in P:
			P[pred_class] = []
			T_view[pred_class] = []
			T_bbox[pred_class] = []
		P[pred_class].append(pred_prob)
		if len(gt_new) == 0:
			T_bbox[pred_class].append(0)
			T_view[pred_class].append(0)
			continue
		iou = roi_helpers.overlap_mAVP(np.array(gt_bbox),pred_box)
		if max(iou) >= 0.5 and not(gt_new[np.argmax(iou)]['bbox_matched']):
			gt_new[np.argmax(iou)]['bbox_matched'] = True
			T_bbox[pred_class].append(1)
			if discretize(float(pred_box['azimuth']),24) ==  discretize(float(gt_new[np.argmax(iou)]['azimuth']),24):
			# if float(pred_box['azimuth']) ==  discretize(float(gt_new[np.argmax(iou)]['azimuth']),24):
				gt_new[np.argmax(iou)]['view_matched'] = True
				T_view[pred_class].append(1)
			else:
				T_view[pred_class].append(0)

		else:
			T_bbox[pred_class].append(0)
			T_view[pred_class].append(0)
	return T_view, T_bbox, P

def VOCap(rec,prec):
	mrec = np.concatenate([np.concatenate([np.array([[0]]),rec],axis=1),np.array([[1]])],axis=1)
	mpre = np.concatenate([np.concatenate([np.array([[0]]),prec],axis=1),np.array([[0]])],axis=1)
	for ii in range(mpre.size-2,-1,-1):
		mpre[0,ii] = max(mpre[0,ii],mpre[0,ii+1])
	ii = np.where((mrec[0,1:] != mrec[0,:-1]) == True)[0]+1
	# sio.savemat('python_mat.mat',{'mpre_p':mpre,'mrec_p':mrec,'ii':ii})
	ap =sum((mrec[0,ii]-mrec[0,ii-1])*mpre[0,ii])
	return ap

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc"),
parser.add_option("--input_train_file", dest="input_train_file", help="if there is a pickle file for train data.",default='pickle_data/train_data_Wflip_all.pickle' )

(options, args) = parser.parse_args()

# if not options.test_path:   # if filename is not given
# 	parser.error('Error: path to test data must be specified. Pass --path to command line')



if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

config_output_filename = options.config_filename

with open(config_output_filename, 'r') as f_in:
	C = pickle.load(f_in)

## define all the paths
test_From_File = False
use_NN = False
curr_path = os.getcwd()
test_path = os.path.join(curr_path,'VOCdevkit/VOC3D')
# weight_name = 'Massa'
weight_name = 'model_FC_weight_leaky_best'
# weight_name = 'model_trip_real_only_aeroplane_best'
C.model_path = os.path.join(curr_path,'models/{}.hdf5'.format(weight_name))

## create txt files
eval_folder = os.path.join(curr_path,'Evaluation')
if not(os.path.exists(eval_folder)):
	os.mkdir(eval_folder)
try:
	eval_folder = os.path.join(eval_folder,weight_name)
	os.mkdir(eval_folder)
except:
	pass
test_cls = ['aeroplane','bicycle','boat','bus','car','chair','diningtable','motorbike','sofa','train', 'tvmonitor']



# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

if os.path.exists('pickle_data/mAVP_test_file.pickle'):
	with open('pickle_data/mAVP_test_file.pickle') as f:
		all_imgs, _, _ = pickle.load(f)
else:
	all_imgs, _, _ = get_data(test_path)
test_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
test_imgs.sort(key= lambda k: k['filepath'])




def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	fx = width/float(new_width)
	fy = height/float(new_height)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img, fx, fy


class_mapping = C.class_mapping

def pred2bins(az_pred):
    '''
    find the bin for every quntization by sum
    '''
    sum_az = np.cumsum(az_pred)
    pyr_bins = np.zeros([1,4])
    vnum = [4, 8, 16, 24]
    for ii in range(len(vnum)):
        tmp_azimuth = np.concatenate(([0], np.linspace((360. / (vnum[ii] * 2)), 360. - (360. / (vnum[ii] * 2)), vnum[ii])),
                                 axis=0)
        sum_vec = np.diff([sum_az[i] for i in tmp_azimuth.astype(int)])
        sum_vec[0] += sum_az[359] - sum_vec[-1]
        pyr_bins[0,ii] = np.argmax(sum_vec) + 1 #becuse the bins starts at 1
    return pyr_bins

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)
def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	x = x- np.min(x)
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

inv_class_mapping = {v: k for k, v in class_mapping.iteritems()}
print(inv_class_mapping)
class_to_color = {inv_class_mapping[v]: np.random.randint(0, 255, 3) for v in inv_class_mapping}
C.num_rois = int(options.num_rois)
T_view, T_bbox = {}, {}
P = {}
txt_files = {}
st = time.time()
txt_files['results'] = open(os.path.join(eval_folder, "results.txt"), 'w')
## if want to run the network
if not(test_From_File):
	for cls in test_cls:
		txt_files[cls] = open(os.path.join(eval_folder, "{}.txt".format(cls)), 'w')

	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
		input_shape_features = (1024, None, None)
	else:
		input_shape_img = (None, None, 3)
		input_shape_features = (None, None, 1024)


	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))
	feature_map_input = Input(shape=input_shape_features)

	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)

	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn = nn.rpn(shared_layers, num_anchors)

	classifier,inner = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping))

	inner = Lambda(lambda x: tf.nn.l2_normalize(x, dim=2))(inner)

	model_rpn = Model(img_input, rpn)

	model_classifier = Model([feature_map_input, roi_input], classifier)
	model_inner = Model([feature_map_input, roi_input], inner)

	model_rpn.load_weights(C.model_path, by_name=True)
	model_classifier.load_weights(C.model_path, by_name=True)
	model_inner.load_weights(C.model_path, by_name=True)

	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')

	## get GT for all az for single cls
	if use_NN and not(os.path.exists('pickle_data/{}_NN.pickle'.format(weight_name))):
		test_cls_NN = 'aeroplane'
		gt_cls_num = class_mapping[test_cls_NN]
		bbox_threshold_orig = 0.6
		feature_az = []
		sorted_path = options.input_train_file
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
					im_file.append(trip_data[test_cls_NN][ii][jj])
					ind.append(ii)
				except:
					if jj == 0:
						print('no azimuth {}'.format(ii))
		data_gen_train = data_generators.get_anchor_gt(im_file, [], C, K.image_dim_ordering(), mode='test')
		azimuth_dict = []
		inner_NN = []
		for tt in range(len(ind)):
			try:
				if tt%100 == 0:
					print ('worked on {}/{}'.format(tt,len(ind)))
				# print ('im num {}'.format(good_img))
				X, Y, img_data = next(data_gen_train)

				[Y1, Y2, F] = model_rpn.predict_on_batch(X)

				R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,
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

					[P_cls, P_regr, P_view] = model_classifier.predict([F, ROIs])
					inner_f = model_inner.predict([F, ROIs])
					# oo = model_classifier_only.predict([F, ROIs])


					for ii in range(len(sel_samples)):

						if np.max(P_cls[0, ii, :]) < bbox_threshold_orig or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
							continue

						## use gt class
						cls_num = gt_cls_num

						cls_name = inv_class_mapping[cls_num]
						cls_view = P_view[0, ii, 360 * cls_num:360 * (cls_num + 1)]


						# azimuths[cls_name].append(np.argmax(cls_view, axis=0))
						inner_NN.append(inner_f[0,ii,:])
						azimuth_dict.append(img_data['bboxes'][0]['azimuth'])
			except:
				print('failed on az {}'.format(img_data['bboxes'][0]['azimuth']))
		## calculating some mean feature map for every az
		with open('pickle_data/{}_NN.pickle'.format(weight_name),'w') as f:
			pickle.dump([inner_NN,azimuth_dict],f)
			print('saved PICKLE')
		neigh = KNeighborsClassifier(n_neighbors=1)
		neigh.fit(inner_NN, azimuth_dict)
	elif use_NN and os.path.exists('pickle_data/{}_NN.pickle'.format(weight_name)):
         with open('pickle_data/{}_NN.pickle'.format(weight_name)) as f:
            inner_NN, azimuth_dict = pickle.load(f)
            print('loaded NN data for current weight')
         neigh = KNeighborsClassifier(n_neighbors=1)
         neigh.fit(inner_NN, azimuth_dict)

	tsne_data ={}
	for idx, img_data in enumerate(test_imgs):
		if idx %50 == 0:
			print('{}/{}'.format(idx,len(test_imgs)))

		filepath = img_data['filepath']

		img = cv2.imread(filepath)

		X, fx, fy = format_img(img, C)

		if K.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		[Y1, Y2,F] = model_rpn.predict(X)

		R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		# apply the spatial pyramid pooling to the proposed regions
		bboxes = {}
		probs = {}
		azimuths,az_total = {},{}
		inner_res = {}

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

			[P_cls, P_regr,P_view] = model_classifier.predict([F ,ROIs])
			if use_NN:
				inner_out = model_inner.predict([F, ROIs])

			for ii in range(P_cls.shape[1]):

				if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = inv_class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []
					azimuths[cls_name] = []
					inner_res[cls_name] = []
					az_total[cls_name] = np.empty((0,360))

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
					tx /= C.classifier_regr_std[0]
					ty /= C.classifier_regr_std[1]
					tw /= C.classifier_regr_std[2]
					th /= C.classifier_regr_std[3]
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))
				azimuths[cls_name].append(np.argmax(P_view[0, ii, 360*cls_num:360*(cls_num+1)]))
				az_total[cls_name] = np.append(az_total[cls_name],[softmax(P_view[0, ii, 360*cls_num:360*(cls_num+1)]) * np.max(P_cls[0, ii, :])],axis = 0)
				if use_NN:
					inner_res[cls_name].append(inner_out[0,ii,:])
		all_dets = []

		for key in bboxes:
			bbox = np.array(bboxes[key])
			prob = np.array(probs[key])
			if use_NN:
				inner_result = np.array(inner_res[key])
				# img = draw_bbox(img,bbox, prob, azimuth, ratio)
				azimuth = neigh.predict(inner_result)
			else:
				azimuth = np.array(azimuths[key])
				az_tot = np.argmax(np.array(az_total[key]))
			new_boxes, new_probs,new_azimuth,new_total= roi_helpers.non_max_suppression_fast(bbox,prob,azimuth,az_total[key], overlap_thresh=0.5,use_az=True,use_total=True)
			# new_azimuth = np.argmax(new_total,axis=1).tolist()
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk, :]
				det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk],'azimuth':new_azimuth[jk]}
				if key in test_cls:
					file_name = os.path.splitext(os.path.split(filepath)[1])[0]
					txt_files[key].write('{} {} {} {} {} {} {}\n'.format(file_name,int(x1*fx),int(y1*fy),int(x2*fx),int(y2*fy),new_probs[jk],new_azimuth[jk]))
				all_dets.append(det)

	for key in test_cls:
		txt_files[key].close()

count ={}
for cls_txt in test_cls:
# for cls_txt in ['aeroplane']:
	count[cls_txt] = 0
	with open(os.path.join(eval_folder,'{}.txt'.format(cls_txt)),'r') as f:
		text = f.readlines()
		text = [x.split() for x in text]
		## find the data that match this line
	for ii in range(len(test_imgs)):
		good_bbox = [x for x in test_imgs[ii]['bboxes'] if x['class'] == cls_txt]
		count[cls_txt] = count[cls_txt] + len(good_bbox) - sum([x['difficult'] for x in good_bbox])

	old_name =''
	all_dets = [{'x1': int(text[0][1]), 'x2': int(text[0][3]), 'y1': int(text[0][2]), 'y2': int(text[0][4]),
			   'class': cls_txt, 'prob': float(text[0][5]),
			   'azimuth': int(text[0][6])}]

	for ii in range(1,len(text)-1):
		det = {'x1': int(text[ii][1]), 'x2': int(text[ii][3]), 'y1': int(text[ii][2]), 'y2': int(text[ii][4]),
			   'class': cls_txt, 'prob': float(text[ii][5]),
			   'azimuth': int(text[ii][6])}
		if text[ii][0] == text[ii-1][0]:
			all_dets.append(det)
		else:
			idx = [text[ii-1][0] in x['filepath'] for x in test_imgs].index(True)
			t_view, t_bbox, p = get_mAVP(all_dets, test_imgs[idx]['bboxes'], (1, 1),key=cls_txt)
			for key in t_view.keys():
				if key not in T_view:
					T_view[key] = []
					T_bbox[key] = []
					P[key] = []
				T_bbox[key].extend(t_bbox[key])
				T_view[key].extend(t_view[key])
				P[key].extend(p[key])
			all_dets = [det]
		if ii == len(text)-1 and text[ii][0] == text[ii-1][0]:
			idx = [text[ii][0] in x['filepath'] for x in test_imgs].index(True)
			t_view, t_bbox, p = get_mAVP(all_dets, test_imgs[idx]['bboxes'], (1, 1),key=cls_txt)
			for key in t_view.keys():
				if key not in T_view:
					T_view[key] = []
					T_bbox[key] = []
					P[key] = []
				T_bbox[key].extend(t_bbox[key])
				T_view[key].extend(t_view[key])
				P[key].extend(p[key])

all_aps = []
all_avps = []

for key in test_cls:
# for key in ['aeroplane']:
	idexs = sorted(range(len(P[key])), key=lambda x: P[key][x], reverse=True)
	precision = np.zeros([1, len(idexs)])
	accuracy = np.zeros([1, len(idexs)])
	recall_bbox = np.zeros([1, len(idexs)])
	recall_view = np.zeros([1, len(idexs)])
	num_correct = 0.
	num_correct_view = 0.
	for num_positive,id in enumerate(idexs):
		num_correct = num_correct + T_bbox[key][id]
		if num_positive+1 != 0:
			precision[0,num_positive] = num_correct / (num_positive+1)
		else:
			precision[0,num_positive] = 0.


		num_correct_view = num_correct_view + T_view[key][id]
		if num_positive+1 != 0:
			accuracy[0,num_positive] = num_correct_view / (num_positive+1)
		else:
			accuracy[0,num_positive] = 0.

		recall_bbox[0,num_positive] = num_correct / count[key]
		recall_view[0,num_positive] = num_correct_view / count[key]

	## my new ap calculation, based on Render4CNN
	avp = VOCap(rec=recall_bbox,prec=accuracy)
	ap = VOCap(rec=recall_bbox,prec=precision)

	## old ap calculation
	# ap = average_precision_score(T_bbox[key], P[key])
	# avp = average_precision_score(T_view[key], P[key])
	print('{} AP: {}'.format(key, ap))
	print('{} AVP: {}'.format(key, avp))
	txt_files['results'].write('{} AP: {}\n'.format(key, ap))
	txt_files['results'].write('{} AVP: {}\n'.format(key, avp))
	all_aps.append(ap)
	all_avps.append(avp)
print('mAP = {}'.format(np.mean(np.array(all_aps))))
print('mVAP = {}'.format(np.mean(np.array(all_avps))))
txt_files['results'].write('mAP = {}\n'.format(np.mean(np.array(all_aps))))
txt_files['results'].write('mVAP = {}\n'.format(np.mean(np.array(all_avps))))
txt_files['results'].close()	# print(T)
	# print(P)
print('Elapsed time = {}'.format(time.time() - st))
