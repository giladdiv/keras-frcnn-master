import numpy as np
import cv2
from PIL import Image


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
## read the training data from pickle file or from annotations
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))
	return (real_x1, real_y1, real_x2 ,real_y2)

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C,rgb = False):
    """ formats the image channels based on config """
    if not(rgb):
        img = img[:, :, (2, 1, 0)] ## not used because imageio read as RGB and not BGR like cv2
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def draw_bbox(img, bbox, prob, azimuth, ratio,class_mapping,key):
    # new_boxes, new_probs, new_az = roi_helpers.non_max_suppression_fast(bbox, prob, azimuth, overlap_thresh=0.3,use_az=True)
    new_boxes = bbox
    new_az = azimuth
    new_probs = prob
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    for jk in range(new_boxes.shape[0]):
        (x1, y1, x2, y2) = new_boxes[jk, :]

        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

        cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                      (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)
        # cv2.rectangle(img,(bbox_gt['x1'], bbox_gt['y1']), (bbox_gt['x2'], bbox_gt['y2']), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

        # textLabel = '{}: {},azimuth : {}'.format(key,int(100*new_probs[jk]),new_az[jk])
        textLabel = 'azimuth : {}'.format(new_az[jk])

        # all_dets.append((key, 100 * new_probs[jk]))

        (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        textOrg = (real_x1, real_y1 + 15)

        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    return img


def format_img(img, C,rgb = False):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C,rgb=rgb)
    return img, ratio


def display_image(img,title_id):
    cv2.putText(img, str(title_id), (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    # img1 = img[:, :, (2, 1, 0)]
    img1=img
    im = Image.fromarray(img1.astype('uint8'), 'RGB')
    im.show()