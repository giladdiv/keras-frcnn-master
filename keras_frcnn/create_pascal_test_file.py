import os
pascal_path = '/home/gilad/ssd/PASCAL3D+_release1.1/PASCAL/VOCdevkit/VOC2012/ImageSets/Main'
old_cls = ['aeroplane','bicycle','boat','bottle','bus','car','chair','diningtable','motorbike','sofa','train', 'tvmonitor']
for cls in old_cls:
    f_temp = open(os.path.join(pascal_path,cls+'_test_only.txt'),'wb')
    with open(os.path.join(pascal_path,cls+'_val.txt'))  as f:
        for line in f:
            if line[-3] != '-':
                f_temp.write(line.split()[0]+'\n')
    f_temp.close()