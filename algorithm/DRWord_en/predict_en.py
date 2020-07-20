import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append('/home/wsw/workplace/deeplearning/DRWord/algorithm/DRWord_en/')

import locality_aware_nms as nms_locality
#import lanms
from bktree import BKTree, levenshtein, list_words

from module2 import Backbone_branch2, Recognition_branch2, RoI_rotate2
from data_provider2.data_utils2 import restore_rectangle, ground_truth_to_word2
FLAGS = tf.app.flags.FLAGS
detect_part2 = Backbone_branch2.Backbone2(is_training=False)
roi_rotate_part2 = RoI_rotate2.RoIRotate2()
recognize_part2 = Recognition_branch2.Recognition2(is_training=False)
font = cv2.FONT_HERSHEY_SIMPLEX

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes

def get_project_matrix_and_width(text_polyses, target_height=8.0):
    project_matrixes = []
    box_widths = []
    filter_box_masks = []
    # max_width = 0
    # max_width = 0

    for i in range(text_polyses.shape[0]):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 4
        #print(text_polyses[i])

        #the input of cv2.minAreaRect must be float32, but the intial (x,y) are float64
        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]] ,dtype=np.float32))
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

        if box_w <= box_h:
            box_w, box_h = box_h, box_w

        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 8)

        width_box = math.ceil(8 * box_w / box_h)
        width_box = int(min(width_box, 128)) # not to exceed feature map's width
        # width_box = int(min(width_box, 512)) # not to exceed feature map's width
        """
        if width_box > max_width: 
            max_width = width_box 
        """
        mapped_x2, mapped_y2 = (width_box, 0)
        # mapped_x3, mapped_y3 = (width_box, 8)

        src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
        dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])
        affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        affine_matrix = affine_matrix.flatten()

        # project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        # project_matrix = project_matrix.flatten()[:8]

        project_matrixes.append(affine_matrix)
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]
def bktree_search(bktree, pred_word, dist=5):
    return bktree.query(pred_word, dist)

def predicten(image_path, return_dic):
    checkpoint_path = '/home/wsw/workplace/deeplearning/DRWord/algorithm/DRWord_en/checkpoint/'

    #是否使用GPU：-1不是用；0使用GPU0
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with tf.Graph().as_default(): #tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        input_feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 32], name='input_feature_map')
        input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
        input_box_mask = []
        input_box_mask.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_0'))
        input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')

        input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        shared_feature, f_score, f_geometry = detect_part2.model2(input_images)
        pad_rois = roi_rotate_part2.roi_rotate_tensor_pad2(input_feature_map, input_transform_matrix, input_box_mask, input_box_widths)
        recognition_logits = recognize_part2.build_graph2(pad_rois, input_box_widths)
        _, dense_decode = recognize_part2.decode2(recognition_logits, input_box_widths)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        im_fn_list = []

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list.append(image_path)

            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                # im_resized_d, (ratio_h_d, ratio_w_d) = resize_image_detection(im)

                shared_feature_map, score, geometry = sess.run([shared_feature, f_score, f_geometry], feed_dict={input_images: [im_resized]})
                
                boxes = detect(score_map=score, geo_map=geometry)

                txt_result = []
                im_result = im

                if boxes is not None and boxes.shape[0] != 0:
                    input_roi_boxes = boxes[:, :8].reshape(-1, 8)
                    recog_decode_list = []
                    # Here avoid too many text area leading to OOM
                    for batch_index in range(input_roi_boxes.shape[0] // 32 + 1): # test roi batch size is 32
                        start_slice_index = batch_index * 32
                        end_slice_index = (batch_index + 1) * 32 if input_roi_boxes.shape[0] >= (batch_index + 1) * 32 else input_roi_boxes.shape[0]
                        tmp_roi_boxes = input_roi_boxes[start_slice_index:end_slice_index]

                        boxes_masks = [0] * tmp_roi_boxes.shape[0]
                        transform_matrixes, box_widths = get_project_matrix_and_width(tmp_roi_boxes)
                        # max_box_widths = max_width * np.ones(boxes_masks.shape[0]) # seq_len
                    
                        # Run end to end
                        recog_decode = sess.run(dense_decode, feed_dict={input_feature_map: shared_feature_map, input_transform_matrix: transform_matrixes, input_box_mask[0]: boxes_masks, input_box_widths: box_widths})
                        recog_decode_list.extend([r for r in recog_decode])

                    # Preparing for draw boxes
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                    if len(recog_decode_list) != boxes.shape[0]:
                        return txt_result, im_result

                    
                    for i, box in enumerate(boxes):
                        # to avoid submitting errors
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                            continue
                        recognition_result = ground_truth_to_word2(recog_decode_list[i])
                        txt_result.append(recognition_result)
                        # Draw bounding box
                        im_result = cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                            
                else:
                    im_result = im
                    txt_result = []
    #tf.reset_default_graph()
            return_dic['im_result_en'] = im_result
            return_dic['txt_result_en'] = txt_result
            #return im_result, txt_result

if __name__ == '__main__':
    im_result, txt_result = predicten('/home/wsw/workplace/deeplearning/DRWord/algorithm/test/img_55.jpg')
    print(txt_result)
    print(type(im_result))
    cv2.imwrite('/home/wsw/workplace/deeplearning/DRWord/algorithm/tset1.jpg', im_result)

