import os
import cv2
import pickle
import xml.etree.ElementTree as ET
import numpy as np
from read_mat import *

def get_data(input_path):
	all_imgs = []
	test_cls = ['aeroplane','bicycle','boat','bus','car','chair','diningtable','motorbike','sofa','train', 'tvmonitor']

	classes_count = {}

	class_mapping = {}

	visualise = False

	data_paths = [os.path.join(input_path,s) for s in ['PASCAL/VOCdevkit/VOC2012']]
	

	print('Parsing annotation files')

	for data_path in data_paths:

		annot_path = os.path.join(data_path, 'Annotations')
		imgs_path = os.path.join(data_path, 'JPEGImages')
		imgsets_path_trainval = os.path.join(data_path, 'ImageSets','Main','val.txt')
		imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','test.txt')

		trainval_files = []
		test_files = []
		try:
			with open(imgsets_path_trainval) as f:
				for line in f:
					trainval_files.append(line.strip() + '.jpg')
		except Exception as e:
			print(e)

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
		
		annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
		idx = 0
		for annot in annots:
			try:
				idx += 1

				et = ET.parse(annot)
				element = et.getroot()

				element_objs = element.findall('object')
				element_filename = element.find('filename').text
				element_width = int(element.find('size').find('width').text)
				element_height = int(element.find('size').find('height').text)

				if len(element_objs) > 0:
					annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
									   'height': element_height, 'bboxes': []}

					if element_filename in trainval_files:
						annotation_data['imageset'] = 'trainval'
					elif element_filename in test_files:
						annotation_data['imageset'] = 'test'
					else:
						continue
						annotation_data['imageset'] = 'train'

				for idx,element_obj in enumerate(element_objs):
					class_name = element_obj.find('name').text
					if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1
					if class_name in test_cls:
						az_flag =True
					else:
						az_flag = False
					if class_name not in class_mapping:
						class_mapping[class_name] = len(class_mapping)

					if az_flag:
						annot = os.path.join(input_path,'Annotations/{}_pascal/{}.mat'.format(class_name,os.path.splitext(element_filename)[0]))
						az_element = loadmat(annot)['record']
						az_element_obj = az_element['objects{}'.format(idx)]
						view = az_element_obj['viewpoint']
						if view['distance'] == 0:
							azimuth = int(view['azimuth_coarse'])%360
						else:
							azimuth = float(view['azimuth'])%360
					obj_bbox = element_obj.find('bndbox')
					x1 = int(round(float(obj_bbox.find('xmin').text)))
					y1 = int(round(float(obj_bbox.find('ymin').text)))
					x2 = int(round(float(obj_bbox.find('xmax').text)))
					y2 = int(round(float(obj_bbox.find('ymax').text)))
					difficulty = int(element_obj.find('difficult').text) == 1
					if az_flag:
						tmp = {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty,'azimuth data': True,'azimuth':azimuth}
					else:
						tmp = {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty,'azimuth data': False}
					annotation_data['bboxes'].append(tmp)
				all_imgs.append(annotation_data)

				if visualise:
					img = cv2.imread(annotation_data['filepath'])
					for bbox in annotation_data['bboxes']:
						cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
									  'x2'], bbox['y2']), (0, 0, 255))
					cv2.imshow('img', img)
					cv2.waitKey(0)

			except Exception as e:
				print(e)
				continue
	with open('pickle_data/mAVP_test_file.pickle','w') as f:
		pickle.dump([all_imgs, classes_count, class_mapping],f)

	return all_imgs, classes_count, class_mapping
