# saves all the utils of the ssd network
import tensorflow as tf
import tensorflow.contrib.slim as slim
# import ndcg_recsys

def base_net_arg_scope():
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME') as sc:
                return sc


def base_net(inputs,
            num_classes=400,
            rois=None,
            bbox_regression=False,
            is_training=True,
            dropout_keep_prob=0.5,
            reuse=None,
            scope='ssd_300_vgg'):
    with slim.arg_scope(base_net_arg_scope()):
        end_points = {}
        with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
            # Original VGG-16 blocks.
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            end_points['block1'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # Block 2.
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            end_points['block2'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # Block 3.
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            end_points['block3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # Block 4.
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            end_points['block4'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # Block 5.
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            end_points['block5'] = net
            net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

            # Block 6: let's dilate the hell out of it!
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
            end_points['block6'] = net
            # Block 7: 1x1 conv.
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            tf.summary.histogram('block7_hist', net)
            end_points['block7'] = net

            # Block 8.
            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                tf.summary.histogram('block8', net)
            end_points[end_point] = net

            # prediction part
            end_point = 'prediction'
            with tf.variable_scope(end_point):
                net = end_points['block8']
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')
                tf.summary.histogram('block8_dp', net)
                net = slim.conv2d(net, 1024, [9, 9], padding='VALID', scope='conv9x9')
                tf.summary.histogram('pred9x9', net)
                net = tf.reduce_mean(net, axis=[1, 2], keep_dims=True)
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout2')
                net = slim.conv2d(net, num_classes, [1, 1], scope='fc')
            end_points[end_point] = net
            prediction = tf.squeeze(net, [1, 2], name='squeezed')

            if not (rois is None):
                #todo: assume the batch size is 1
                end_point = 'fast_rcnn'
                with tf.variable_scope(end_point):
                    net = end_points['block8']
                    net = tf.image.resize_images(net, [32, 32])
                    rois_yxyx = rois[:, [1, 0, 3, 2]]
                    net = tf.image.crop_and_resize(net, boxes=rois_yxyx, crop_size=[3,3], box_ind=tf.zeros(shape=[num_classes], dtype=tf.int32))
                    # this dropout is not good enough for limited data
                    #TODO: using 1024 is for pairwise-frcnn-dp(dropout + large)
                    #TODO: there are different versions: dp+1024/128/1024+dp+128+dp
                    #TODO: update2: remove the complex structure, just keep the 128D
                    # is using 128 directly, it means using pairwise-frcnn
                    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout_cls')
                    net = slim.conv2d(net, 1024, [3, 3], padding='SAME', scope='conv3x3')
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout_cls')
                    net = slim.conv2d(net, 128, [3, 3], padding='VALID', scope='conv1x1_2')
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout_cls')
                    net = slim.conv2d(net, 1, [1, 1], padding='SAME', scope='conv1x1_1')
                end_points[end_point] = net
                cls = tf.squeeze(net, [1, 2], name='squeezed_cls')
                cls = tf.reshape(cls, [1, -1])
            else:
                cls = None

            if bbox_regression:
                #todo: assume the batch size is 1
                end_point = 'bbox_regression'
                with tf.variable_scope(end_point):
                    net = end_points['block8']
                    net = tf.image.resize_images(net, [32, 32])
                    rois_yxyx = rois[:, [1, 0, 3, 2]]
                    net = tf.image.crop_and_resize(net, boxes=rois_yxyx, crop_size=[3,3], box_ind=tf.zeros(shape=[num_classes], dtype=tf.int32))
                    # net = slim.conv2d(net, 1024, [3, 3], padding='SAME', scope='conv3x3')
                    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout_breg')
                    net = slim.conv2d(net, 128, [3, 3], padding='VALID', scope='conv1x1_2')
                    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout_breg')
                    net = slim.conv2d(net, 4, [1, 1], padding='SAME', scope='conv1x1_1')
                end_points[end_point] = net
                regress = tf.squeeze(net, [1, 2], name='squeezed_regress')
            else:
                regress = None

            return prediction, cls, regress, end_points


def swap_correct(logits, labels, batch_size, n_anchors=895, scope=None):
    with tf.variable_scope(scope, 'swap_correct'):
        logits_offset0 = logits[:, 0:n_anchors-1]
        logits_offset1 = logits[:, 1:n_anchors]

        labels_offset0 = labels[:, 0:n_anchors-1]
        labels_offset1 = labels[:, 1:n_anchors]

        logits_diff = logits_offset0 - logits_offset1
        labels_diff = labels_offset0 - labels_offset1

        correct_mask = tf.greater(tf.multiply(logits_diff, labels_diff), 0)
        n_corrects = tf.reduce_sum(tf.cast(correct_mask, tf.float32))
        avg_correct = n_corrects / n_anchors / batch_size

        return avg_correct


def mean_pairwise_squared_error(logits, gclasses, alpha=1., scope=None):
    """continious pairwise loss:
    """
    with tf.variable_scope(scope, 'mean_pairwise_square_error'):
        total_loss = tf.losses.mean_pairwise_squared_error(gclasses, logits, weights=alpha)
    return total_loss


def bbox_reg_loss(pred_xywh, labels, anchor_xywh, nearest_xywh):
    pred_xy = pred_xywh[:, 0:2]
    pred_wh = pred_xywh[:, 2:4]
    sig_pred_xy = tf.sigmoid(pred_xy)
    exp_pred_wh = tf.exp(pred_wh)
    anchor_xy = anchor_xywh[:, 0:2]
    anchor_wh = anchor_xywh[:, 2:4]
    nearest_xy = nearest_xywh[:, 0:2]
    nearest_wh = nearest_xywh[:, 2:4]
    diff_xy = sig_pred_xy - 0.5 + anchor_xy - nearest_xy
    diff_wh = exp_pred_wh - nearest_wh / anchor_wh
    loss = tf.multiply(tf.reduce_sum(abs_smooth(diff_xy), 1, keep_dims=True) + tf.reduce_sum(abs_smooth(diff_wh), 1, keep_dims=True), labels)

    return tf.reduce_sum(loss)


def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

