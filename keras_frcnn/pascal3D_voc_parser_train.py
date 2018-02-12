
import os
import cv2
import xml.etree.ElementTree as ET
import itertools
from read_mat import *
from PIL import Image
import numpy as np
import time
import pickle
from time import sleep
import copy
from keras_frcnn.Quaternion import Quat

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

@timing

def get_data():
	base_path = os.getcwd() + '/../VOCdevkit'
	# data_phase = "test"
	cls = "aeroplane"
	# base_path = '/home/gilad/ssd/keras-frcnn-master/VOCdevkit'
	all_imgs = []
	data_by_cls ={}
	classes_count = {}
	class_mapping = {}
	azimuth_dist ={}


	flip_flag = True
	write_flip = False
	visualise = False
	data_paths = [os.path.join(base_path,s) for s in ['VOC3D']]



	print('Parsing annotation files')
	skip_bbox = 0
	for data_path in data_paths:
		# for data_type in ['imagenet','pascal']:
		for data_type in ['pascal','imagenet']:
			annot_path = os.path.join(data_path, 'Annotations')
			if data_type =='pascal':
				imgs_path = os.path.join(data_path, 'JPEGImages')
			else:
				imgs_path = os.path.join(data_path, 'imagenet_images')
			imgsets_path_trainval = os.path.join(data_path, 'ImageSets','Main','trainval.txt')
			imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','all_test.txt')
			annots = []
			trainval_files = []
			test_files = []
			# try:
			# 	with open(imgsets_path_trainval) as f:
			# 		for line in f:
			# 			trainval_files.append(line.strip() + '.jpg')
			# except Exception as e:
			# 	print(e)
			#
			try:
				with open(imgsets_path_test) as f:
					for line in f:
						test_files.append(line.strip() + '.jpg')
			except Exception as e:
				if data_path[-7:] == 'VOC2012':
					# this is expected, most pascal voc distibutions dont have the test.txt file
					pass
				else:
					print(e)
			## to read all the data
			ann_sub_folders = [x for x in os.listdir(annot_path) if x.endswith(data_type)]
			for ii in range(len(ann_sub_folders)):
				annots.append([os.path.join(annot_path,ann_sub_folders[ii],s) for s in os.listdir(os.path.join(annot_path,ann_sub_folders[ii]))])
			annots = list(itertools.chain.from_iterable(annots))

			## to read just one category
			# ann_sub_folders = [x for x in os.listdir(annot_path) if x.endswith(data_type)]
			# for ii in range(len(ann_sub_folders)):
			# 	annots.append([os.path.join(annot_path, ann_sub_folders[ii], s) for s in
			# 				   os.listdir(os.path.join(annot_path, ann_sub_folders[ii]))])
			# annots = list(itertools.chain.from_iterable(annots))
			idx = 0
			for id,annot in enumerate(annots):
				try:
					if id%500 == 0:
						print('finished {} images'.format(id))
					idx += 1

					# et = ET.parse(annot)is not python
					# element = et.getroot()
					element = loadmat(annot)['record']
					element_objs_num = element['objects_num']
					element_filename = element['filename']
					element_width = element['size']['width']
					element_height = element['size']['height']

					if element_objs_num > 0:

						## creat flip version of the image
						if element_filename in test_files:
							continue
						else:
							annotation_data = {'filepath': os.path.join(imgs_path, element_filename),
											   'width': element_width,
											   'height': element_height, 'bboxes': [], 'viewpoint': [],'viewpoint_data': False}
							annotation_data_cls = copy.deepcopy(annotation_data)
							annotation_data['imageset'] = 'train'
							if flip_flag :
							# if flip_flag and data_type =='pascal':
								string = annotation_data['filepath']
								img = cv2.imread(string)
								img_lr = cv2.flip(img, 1)
								try:
									ind = string.index('.j')
								except:
									ind = string.index('.J')
								new_path = string[:ind] + '_flip' + string[ind:]
								if write_flip:
									cv2.imwrite(new_path, img_lr)
								annotation_data_lr = {'filepath': new_path,
											   'width': element_width,
											   'height': element_height, 'bboxes': [], 'viewpoint': [],'viewpoint_data': False}
								annotation_data_lr_cls = copy.deepcopy(annotation_data_lr)
					for ii in range(element_objs_num):
						element_obj = element['objects{}'.format(ii)]
						class_name = element_obj['class']
						## add flip###
						if class_name not in classes_count:
							classes_count[class_name] = 1
							azimuth_dist[class_name] = []
							data_by_cls[class_name] = []
						else:
							classes_count[class_name] += 1

						if class_name not in class_mapping:
							class_mapping[class_name] = len(class_mapping)
						if data_type =='pascal':
							obj_bbox = element_obj['bndbox']
							x1 = obj_bbox['xmin']
							y1 = obj_bbox['ymin']
							x2 = obj_bbox['xmax']
							y2 = obj_bbox['ymax']
							difficulty = element_obj['difficult'] == 1
							# annotation_data['bboxes'].append(
							# 	{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
						else:
							obj_bbox = element_obj['bbox']
							x1 = int(obj_bbox[0])
							y1 = int(obj_bbox[1])
							x2 = int(obj_bbox[2])
							y2 = int(obj_bbox[3])
							difficulty = element_obj['difficult'] == 1
							# annotation_data['bboxes'].append(
							# 	{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})

						## read pose_data
						try:
							obj_view = element_obj['viewpoint']

							if obj_view['distance'] == 0:
								skip_bbox += 1
								# print(string)
								continue
								annotation_data['viewpoint_data'] = True
								az = int(obj_view['azimuth_coarse'])%360
								el = int(obj_view['elevation'])%360
								t = int(obj_view['theta'])%360
							else:
								az = int(obj_view['azimuth'])%360
								el = int(obj_view['elevation'])%360
								t = int(obj_view['theta'])%360
							q = Quat([az,el,t])
							curr_bbox = {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty,'azimuth':az,'elevation':el,'tilt':t,'viewpoint_data':True,'quat':q}
							annotation_data_cls['bboxes'] = [curr_bbox]
							data_cls = annotation_data_cls
							annotation_data['bboxes'].append(curr_bbox)
							azimuth_dist[class_name].append(az)
							data_by_cls[class_name].append(data_cls)
							# if flip_flag and data_type =='pascal':
							if flip_flag:
								## x1 and x2 need to be replaced when fliping the image
								annotation_data_lr['viewpoint_data'] = annotation_data['viewpoint_data']
								q = Quat([(360 - az) % 360,el,t])
								curr_bbox ={'class': class_name, 'x1': element_width - x2, 'x2': element_width - x1, 'y1': y1,'y2': y2, 'difficult': difficulty, 'azimuth': (360 - az) % 360, 'elevation': el,
									 'tilt': t, 'viewpoint_data': True, 'quat': q}
								annotation_data_lr['bboxes'].append(curr_bbox)
								annotation_data_lr_cls['bboxes'] = [curr_bbox]
								data_cls = annotation_data_lr_cls
								data_by_cls[class_name].append(data_cls)
						except:
							annotation_data['bboxes'].append({'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty,'azimuth': 0,'elevation': 0,'tilt': 0,'viewpoint_data':False})


					if annotation_data['imageset'] !='test' and  len(annotation_data['bboxes'])!=0:
						all_imgs.append(annotation_data)
						# if flip_flag and data_type =='pascal':
						if flip_flag:
							all_imgs.append(annotation_data_lr)

					if visualise:
						img = cv2.imread(annotation_data['filepath'])
						for bbox in annotation_data['bboxes']:
							cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
										  'x2'], bbox['y2']), (0, 0, 255))
						# cv2.imshow('img', img)
						# cv2.waitKey(0)
						im = Image.fromarray(img.astype('uint8'),'RGB')
						im.show()
						sleep(2)
						im.close()

				except Exception as e:
					print(e)
					continue
			print('there were {} images in {}'.format(len(all_imgs), data_type))
		# with open('data.txt', 'w') as outfile:
		# 	json.dump(all_imgs, outfile,indent=4, sort_keys=True, ensure_ascii=False)
		# with open('data.txt','r') as f2:
		# 	a  =ymal.safe_load(f2)
	name = 'train_data_Wflip_all'
	with open('../pickle_data/{}.pickle'.format(name), 'w') as f:  # Python 3: open(..., 'wb')
		pickle.dump([all_imgs, classes_count, class_mapping], f)

	with open('../azimuth_distibution.pickle', 'w') as f:  # Python 3: open(..., 'wb')
		pickle.dump([azimuth_dist], f)

	for key,item in data_by_cls.items():
		data_by_cls[key] = sorted(item,key =lambda k: k['bboxes'][0]['azimuth'])

	with open('../pickle_data/{}_sorted.pickle'.format(name), 'w') as f:  # Python 3: open(..., 'wb')
		pickle.dump(data_by_cls, f)
	# # Getting back the objects:
	# with open('train_data.pickle') as f:  # Python 3: open(..., 'rb')
	# 	obj0, obj1, obj2 = pickle.load(f)
	print ('skiped {} bboxes'.format(skip_bbox))
	return all_imgs, classes_count, class_mapping
	#
if __name__ == "__main__":
	get_data()