import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

#tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS

def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

class Backbone(object):
    def __init__(self, is_training=True):
        self.is_training = is_training

    def model(self, images, weight_decay=1e-5):
        '''
        define the model, we use slim's implemention of resnet
        '''
        images = mean_image_subtraction(images)

        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(images, is_training=self.is_training, scope='resnet_v1_50')

        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': self.is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                f = [end_points['pool5'], end_points['pool4'],
                     end_points['pool3'], end_points['pool2']]
                for i in range(4):
                    print('Shape of f_{} {}'.format(i, f[i].shape))
                g = [None, None, None, None]
                h = [None, None, None, None]
                num_outputs = [None, 128, 64, 32]
                for i in range(4):
                    if i == 0:
                        h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= 2:
                        g[i] = unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

                # here we use a slightly different way for regression part,
                # we first use a sigmoid to limit the regression range, and also
                # this is do with the angle map
                F_score = slim.conv2d(g[3], 1, 1, activation_fn=None, normalizer_fn=None)
                side_v_code = slim.conv2d(g[3], 2, 1, activation_fn=None, normalizer_fn=None)
                side_v_coord = slim.conv2d(g[3], 4, 1, activation_fn=None, normalizer_fn=None) 
              
                F_geometry = tf.concat([F_score, side_v_code, side_v_coord], axis=-1)

        return g[3], F_geometry


    def dice_coefficient(self, y_true_cls, y_pred_cls,
                         training_mask):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
        union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        tf.summary.scalar('classification_dice_loss', loss)
        return loss



    def loss(self, y_true_cls, y_pred_cls,
             y_true_geo, y_pred_geo,
             training_mask):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param y_true_cls: ground truth of text
        :param y_pred_cls: prediction os text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''
        classification_loss = self.dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
        L_g = L_AABB + 20 * L_theta

        return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss

    def quad_loss(self, y_true, y_pred, training_mask):
        # loss for inside_score
        epsilon = 1e-4
        #得分图损失
        logits = y_pred[:, :, :, :1]
        labels = y_true[:, :, :, :1]
        # balance positive and negative samples in an image
        beta = 1 - tf.reduce_mean(labels)
        # first apply sigmoid activation
        predicts = tf.nn.sigmoid(logits)
        # log +epsilon for stable cal
        #epsilon = 1e-4
        inside_score_loss = tf.reduce_mean(
            -1 * (beta * labels * tf.log(predicts + epsilon) +
                (1 - beta) * (1 - labels) * tf.log(1 - predicts + epsilon)))
        lambda_inside_score_loss = 4
        inside_score_loss *= lambda_inside_score_loss

        # loss for side_vertex_code
        #边界像素损失
        vertex_logits = y_pred[:, :, :, 1:3]
        vertex_labels = y_true[:, :, :, 1:3]
        vertex_beta = 1 - (tf.reduce_mean(y_true[:, :, :, 1:2])
                        / (tf.reduce_mean(labels) + epsilon))
        vertex_predicts = tf.nn.sigmoid(vertex_logits)
        pos = -1 * vertex_beta * vertex_labels * tf.log(vertex_predicts + epsilon)
        neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * tf.log(1 - vertex_predicts + epsilon)
        positive_weights = tf.cast(tf.equal(y_true[:, :, :, 0], 1), tf.float32)
        side_vertex_code_loss = \
            tf.reduce_sum(tf.reduce_sum(pos + neg, axis=-1) * positive_weights) / (
                    tf.reduce_sum(positive_weights) + epsilon)
        lambda_side_vertex_code_loss = 1  
        side_vertex_code_loss *= lambda_side_vertex_code_loss

        # loss for side_vertex_coord delta
        #顶点左边损失
        g_hat = y_pred[:, :, :, 3:]
        g_true = y_true[:, :, :, 3:]
        vertex_weights = tf.cast(tf.equal(y_true[:, :, :, 1], 1), tf.float32)
        pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)
        side_vertex_coord_loss = tf.reduce_sum(pixel_wise_smooth_l1norm) / (
                tf.reduce_sum(vertex_weights) + epsilon)
        lambda_side_vertex_coord_loss = 1
        side_vertex_coord_loss *= lambda_side_vertex_coord_loss
    
        model_loss = inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss
        return tf.reduce_mean(model_loss * training_mask)

def smooth_l1_loss(prediction_tensor, target_tensor, weights):
    n_q = tf.reshape(quad_norm(target_tensor), tf.shape(weights))
    diff = prediction_tensor - target_tensor
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    pixel_wise_smooth_l1norm = (tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
        axis=-1) / n_q) * weights
    return pixel_wise_smooth_l1norm


def quad_norm(g_true):
    epsilon = 1e-4
    shape = tf.shape(g_true)
    delta_xy_matrix = tf.reshape(g_true, [-1, 2, 2])
    diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
    square = tf.square(diff)
    distance = tf.sqrt(tf.reduce_sum(square, axis=-1))
    distance *= 4.0
    distance += epsilon
    return tf.reshape(distance, shape[:-1])
