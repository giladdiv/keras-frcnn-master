import os
import  pickle
curr_dir = os.getcwd()
data_path = '/home/gilad/ssd/data'
ann_folder = os.path.join(data_path,'txt_dir')
all_data = {}
data_by_obj = {}
for file_name in os.listdir(ann_folder):
    cls = file_name.split('.')[0]
    print ('work on {}'.format(cls))
    fail = 0
    obj_name = ''
    all_data[cls] = []
    data_by_obj[cls] = {}
    with open(os.path.join(ann_folder,file_name),"r") as f:
        for line in f:
            try:
                tmp = {'filepath': os.path.join(data_path,line.split()[0]),'width': line.split()[13], 'height': line.split()[11], 'bboxes': [], 'viewpoint': []}
                tmp['bboxes'].append({'class': line.split()[1], 'x1': int(float(line.split()[6])), 'x2': int(float(line.split()[6]))+int(float(line.split()[8])), 'y1': int(float(line.split()[7])), 'y2': int(float(line.split()[7])) + int(float(line.split()[9])), 'difficult':False,'azimuth':int(line.split()[2]),'elevation':int(line.split()[3]),'tilt':int(line.split()[4]),'viewpoint_data':True})
                all_data[cls].append(tmp)
                if obj_name == line.split()[0].split('/')[2]:
                    data_by_obj[cls][obj_name].append(tmp)
                else:
                    obj_name = line.split()[0].split('/')[2]
                    data_by_obj[cls][obj_name] = []
                    data_by_obj[cls][obj_name].append(tmp)
            except:
                fail += 1
    print ('failed on {} images'.format(fail))
with open('pickle_data/syn_data.pickle','w') as f:
    pickle.dump(all_data,f)
with open('pickle_data/syn_by_obj_data.pickle','w') as f:
    pickle.dump(data_by_obj,f)