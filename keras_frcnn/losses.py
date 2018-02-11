from keras import backend as K
from keras.objectives import categorical_crossentropy
import numpy as np

if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0
lambda_cls_view = 1.0
lambda_cls_view_weight  = 1.0
# lambda_cls_view_weight
epsilon = 1e-4


def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'th':
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
		else:
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'tf':
			return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))


def class_loss_view(num_classes,roi_num=32):
	def class_loss_view_temp(y_true, y_pred):
		pred = y_pred[0, :, :]
		labels_az = y_true[0,:,:360]
		labels_mask = y_true[0,:,360:-1]
		labels = tf.to_int32(y_true[0,:,-1])

		## find the indicies of the bg
		bg = tf.constant(num_classes-1, dtype=tf.int32)
		indices = tf.where(tf.not_equal(labels, bg))

		zero = tf.constant(0, dtype=tf.float32)
		for i in range(roi_num):
		# if indices.get_shape().as_list()[0] != 0:
		# 	for i in range(indices.get_shape().as_list()[0]):
			indices_mask = tf.where(tf.not_equal(labels_mask[i, :], zero))
			indices_mask = tf.reshape(indices_mask, [-1])
			# indices_mask = tf.Print(indices_mask, [i])
			# indices_mask = tf.Print(indices_mask, [indices_mask])
			if i == 0:
				fc_az_l = tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])
			else:
				fc_az_l = tf.concat(axis=0, values=[fc_az_l, tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])])


		az_tmp = tf.gather_nd(fc_az_l,indices=indices)
		labels_tmp = tf.gather_nd(labels_az,indices=indices)
		y = tf.cond(tf.shape(indices)[0]>0,lambda:lambda_cls_view * tf.reduce_mean(tf.contrib.keras.backend.categorical_crossentropy(output=az_tmp, target=labels_tmp, from_logits=True)),lambda:zero)
		# y = tf.cond(tf.shape(indices)[0]>0,lambda:lambda_cls_view * K.mean(K.binary_crossentropy(labels_tmp,az_tmp)),lambda:zero)
		# y = tf.cond(tf.shape(indices)[0] > 0,lambda: lambda_cls_view *tf.reduce_sum(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_tmp, logits=az_tmp)), lambda: zero)
		# tf.nn.sigmoid_cross_entropy_with_logits
		# else:
		# 	y = zero
		# y = K.sum(y)
		# print(indices.get_shape())
		# y = tf.Print(y,[tf.shape(indices)])
		# y = tf.Print(y,[tf.shape(fc_az_l)])
		# y = tf.Print(y,[tf.shape(az_tmp)])
		# y = tf.Print(y, [indices_mask])
		# y = tf.Print(y, [tf.argmax(labels_tmp, axis=1)])
		# y = tf.Print(y, [tf.argmax(az_tmp, axis=1)])
		# y = tf.Print(y, [loss_temp])
		# y = tf.Print(y, [y])
		# labels_az = tf.Print(labels_az, [tf.shape(fc_az_l)])
		return y
	return class_loss_view_temp

#
def class_loss_view_weight(num_classes,roi_num):
	def class_loss_view_temp(y_true, y_pred):
		pred = y_pred[0, :, :]
		labels_az = y_true[0,:,:360]
		labels_mask = y_true[0,:,360:-1]
		labels = tf.to_int32(y_true[0,:,-1])

		## find the indicies of the bg
		bg = tf.constant(num_classes-1, dtype=tf.int32)
		indices = tf.where(tf.not_equal(labels, bg))

		zero = tf.constant(0, dtype=tf.float32)
		for i in range(roi_num):
			indices_mask = tf.where(tf.not_equal(labels_mask[i, :], zero))
			indices_mask = tf.reshape(indices_mask, [-1])
			if i == 0:
				fc_az_l = tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])
			else:
				fc_az_l = tf.concat(axis=0, values=[fc_az_l, tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])])


		az_tmp = tf.gather_nd(fc_az_l,indices=indices)
		labels_tmp = tf.gather_nd(labels_az,indices=indices)

		y = tf.cond(tf.shape(indices)[0]>0,lambda:lambda_cls_view_weight * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_tmp,logits=az_tmp)),
					lambda:zero)
		# y = K.sum(y)

		# y = tf.Print(y,[tf.shape(az_tmp)])
		# y = tf.Print(y, [tf.argmax(labels_tmp, axis=1)])
		# y = tf.Print(y, [tf.argmax(az_tmp, axis=1)])
		# y = tf.Print(y, [y])
		# labels_az = tf.Print(labels_az, [tf.shape(fc_az_l)])
		return y
	return class_loss_view_temp


def quat_loss(num_classes,roi_num):
	def quat_loss_temp(y_true, y_pred):
		pred = y_pred[0, :, :]
		labels_az = y_true[0,:,:4]
		labels_mask = y_true[0,:,4:-1]
		labels = tf.to_int32(y_true[0,:,-1])

		## find the indicies of the bg
		bg = tf.constant(num_classes-1, dtype=tf.int32)
		indices = tf.where(tf.not_equal(labels, bg))

		zero = tf.constant(0, dtype=tf.float32)
		for i in range(roi_num):
			indices_mask = tf.where(tf.not_equal(labels_mask[i, :], zero))
			indices_mask = tf.reshape(indices_mask, [-1])
			if i == 0:
				fc_az_l = tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])
			else:
				fc_az_l = tf.concat(axis=0, values=[fc_az_l, tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])])


		az_tmp = tf.gather_nd(fc_az_l,indices=indices)
		labels_tmp = tf.gather_nd(labels_az,indices=indices)

		y = tf.cond(tf.shape(indices)[0]>0,lambda:lambda_cls_view_weight * K.mean(K.square(az_tmp- labels_tmp), axis=-1),
					lambda:zero)
		# y = K.sum(y)

		# y = tf.Print(y,[tf.shape(az_tmp)])
		# y = tf.Print(y, [tf.argmax(labels_tmp, axis=1)])
		# y = tf.Print(y, [tf.argmax(az_tmp, axis=1)])
		# y = tf.Print(y, [y])
		# labels_az = tf.Print(labels_az, [tf.shape(fc_az_l)])
		return y
	return quat_loss_temp