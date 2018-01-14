from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class SliceTensor(Layer):
    '''slice tensor layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, cls_num, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.cls_num = cls_num
        self.num_rois = num_rois

        super(SliceTensor, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.data_size = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.data_size = 360

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.data_size
        else:
            return None, self.num_rois, self.data_size

    def call(self, x, mask=None):

        assert(len(x) == 2)

        pred = x[0]
        y = x[1]

        labels_mask = y[0, :, 360:-1]
        labels = tf.to_int32(y[0, :, -1])
        pred = pred[0,:,:]
        # labels_mask = tf.Print(labels_mask,[tf.shape(labels_mask)])

        ## find the indicies of the bg
        bg = tf.constant(self.cls_num - 1, dtype=tf.int32)

        zero = tf.constant(0, dtype=tf.float32)
        for i in range(self.num_rois):
            indices_mask = tf.where(tf.not_equal(labels_mask[i, :], zero))
            indices_mask = tf.reshape(indices_mask, [-1])
            # indices_mask = tf.Print(indices_mask, [i])
            # indices_mask = tf.Print(indices_mask, [indices_mask])
            if i == 0:
                final_output = tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])
            else:
                final_output = tf.concat(axis=0, values=[final_output, tf.reshape(tf.gather(pred[i], indices_mask), [1, -1])])


        final_output = K.reshape(final_output, (1, self.num_rois,self.data_size))

        return final_output
