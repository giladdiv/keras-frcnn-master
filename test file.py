# path_im = '/home/gilad/Dropbox/Project/CNN/viewNet/data_txt/aeroplane_wGT_original.txt'
# with open(path_im,'r') as f:
#     data = f.readlines()
# for line in data:
#     words = line.split()
#     print(words)


import os
start_path = "/home/gilad/ssd/keras-frcnn-master/VOCdevkit/VOC3D/ImageSets/Main"
names = os.listdir(start_path)
ind =[]
for name in names:
    try:
        name.index('only')
        ind.append(name)
    except:
        continue

filenames = [os.path.join(start_path,name) for name in ind]
with open("/home/gilad/ssd/keras-frcnn-master/VOCdevkit/VOC3D/ImageSets/Main/all_test.txt", 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

