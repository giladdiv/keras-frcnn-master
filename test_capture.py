# import cPickle
# from PIL import Image
# import ImageSequence
# import Image
# # import gifmaker
# from images2gif import writeGif
#
# filename ='/home/gilad/bar/actionsAndImagesList1.p'
#
# f = file(filename, 'r')
# tmp = cPickle.load(f)
# frames = []
# for i in range(len(tmp)):
#     frames.append(tmp[i][1])
# f.close()
#
# # im is your original image
#
# # write GIF animation
#
# filename = "my_gif.GIF"
# writeGif(filename, frames, duration=0.2)


import cv2
import os
import sys
import skvideo.io
print(cv2.__version__)
def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    print vidcap.read()
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

video = "/home/gilad/bar/a.mp4"
path_output_dir ='home/gilad/bar'
# video_to_frames(video, path_output_dir)


# cap = skvideo.io.vread(video)
# ret, frame = cap.read()
import pylab
import imageio
filename = "/home/gilad/bar/a.mp4"
vid = imageio.get_reader(filename,  'ffmpeg')
nums = [10, 3000]
# for num in nums:
#     image = vid.get_data(num)
#     fig = pylab.figure()
#     fig.suptitle('image #{}'.format(num), fontsize=20)
#     pylab.imshow(image)
# pylab.show()

try:
    for num, image in enumerate(vid.iter_data()):
        if num % int(vid._meta['fps']):
            continue
        else:
            fig = pylab.figure()
            pylab.imshow(image)
            timestamp = float(num)/ vid.get_meta_data()['fps']
            print(timestamp)
            fig.suptitle('image #{}, timestamp={}'.format(num, timestamp), fontsize=20)
            pylab.show()
except RuntimeError:
    print('something went wrong')
print ('finished')