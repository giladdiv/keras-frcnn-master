import os
import glob

base_path = os.getcwd()
model_path = os.path.join(base_path,'models/model_FC*')
python_file = os.path.join(base_path,'measure_mAVP.py')
for ii in [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(model_path)]:
    print('work on weight {}'.format(ii))
    os.system("python {} -w '{}'".format(python_file,ii))