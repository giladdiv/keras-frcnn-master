import os
import pickle

data_type = 'real'
start_path = '../pickle_data/train_data_Wflip_pascal.pickle'
tmp_ind = start_path.index('.pickle')
sorted_path = start_path[:tmp_ind]+"_sorted"+start_path[tmp_ind:]
ang_path = start_path[:tmp_ind]+"_sorted_Angles"+start_path[tmp_ind:]
if data_type == 'real':
    if os.path.exists(sorted_path):
        print("loading sorted data")
        with open(sorted_path) as f:
            trip_data = pickle.load(f)
else:
    if os.path.exists(start_path):
        print("loading sorted data")
        with open(start_path) as f:
            trip_data = pickle.load(f)

if data_type == 'real':
    new_data = {}
    for key,val in trip_data.items():
        if len(val) != 0:
            new_data[key] = {}
            for single_data in val:
                if single_data['bboxes'][0]['azimuth'] in new_data[key]:
                    new_data[key][single_data['bboxes'][0]['azimuth']].append(single_data)
                else:
                    new_data[key][single_data['bboxes'][0]['azimuth']] = [single_data]
else:
    new_data = {}
    for key,val in trip_data.items():
        if len(val) != 0:
            new_data[key] = {}
            for ii,sub_val in val.items():
                new_data[key][ii] = {}
                for single_data in sub_val:
                    if single_data['bboxes'][0]['azimuth'] in new_data[key][ii]:
                        new_data[key][ii][single_data['bboxes'][0]['azimuth']].append(single_data)
                    else:
                        new_data[key][ii][single_data['bboxes'][0]['azimuth']] = [single_data]

with open(ang_path,'w') as f:
    pickle.dump(new_data,f)




