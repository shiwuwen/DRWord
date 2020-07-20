import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append('/home/wsw/workplace/deeplearning/DRWord/algorithm/DRWord_ch/')

from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import locality_aware_nms as nms_locality
#import lanms
from nms import nms
from module import Backbone_branch, Recognition_branch, RoI_rotate
from data_provider.data_utils import restore_rectangle, ground_truth_to_word
FLAGS = tf.app.flags.FLAGS
detect_part = Backbone_branch.Backbone(is_training=False)
roi_rotate_part = RoI_rotate.RoIRotate()
recognize_part = Recognition_branch.Recognition(is_training=False)
font = cv2.FONT_HERSHEY_SIMPLEX

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))

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

        width_box = math.ceil(8 * box_w / (box_h + 1e-5))
        width_box = int(min(width_box, 128)) # not to exceed feature map's width
        #width_box = int(min(width_box, 512)) # not to exceed feature map's width
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

def predictmtwi(image_path, return_dic):
    checkpoint_path = '/home/wsw/workplace/deeplearning/DRWord/algorithm/DRWord_ch/checkpointmtwi/'

    #是否使用GPU：-1不是用；0使用GPU0
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        
    with tf.Graph().as_default(): #tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        input_feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 32], name='input_feature_map')
        input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
        input_box_mask = []
        input_box_mask.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_0'))
        input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')

        input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        shared_feature, f_geometry = detect_part.model(input_images)
        pad_rois = roi_rotate_part.roi_rotate_tensor_pad(input_feature_map, input_transform_matrix, input_box_mask, input_box_widths)
        recognition_logits = recognize_part.build_graph(pad_rois, input_box_widths)
        _, dense_decode = recognize_part.decode(recognition_logits, input_box_widths)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        im_fn_list = []
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list.append(image_path)
            index = len(im_fn_list)
            for im_fn in im_fn_list:

                #image for draw quad
                im = cv2.imread(im_fn)[:, :, ::-1]       
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                img = image.img_to_array(im_resized)
                img = preprocess_input(img, mode='tf')               

                shared_feature_map, geometry = sess.run([shared_feature, f_geometry], feed_dict={input_images: [img]})
                
                geometry = np.squeeze(geometry, axis=0)
                geometry[:, :, :3] = sigmoid(geometry[:, :, :3])

                pixel_threshold = 0.9
                cond = np.greater_equal(geometry[:, :, 0], pixel_threshold)
                activation_pixels = np.where(cond)

                #boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                quad_scores, boxes = nms(geometry, activation_pixels)
                #print('boxes : ',boxes)

                input_roi_boxes = []
                for score, box in zip(quad_scores, boxes): 
                    if np.amin(score) > 0:
                        #print(type(box))
                        box = box[[0, 3, 2, 1]]
                        input_roi_boxes.append(box)
                input_roi_boxes = np.array(input_roi_boxes)
                #print('input_roi_boxes : ',input_roi_boxes[:])
                
                txt_result = []
                im_result = im
                
                #im_txt = None
                if input_roi_boxes is not None and input_roi_boxes.shape[0] != 0:
                    
                    input_roi_boxes = input_roi_boxes[:, :8].reshape(-1, 8)
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
                        try:
                            recog_decode = sess.run(dense_decode, feed_dict={input_feature_map: shared_feature_map, input_transform_matrix: transform_matrixes, input_box_mask[0]: boxes_masks, input_box_widths: box_widths})
                            recog_decode_list.extend([r for r in recog_decode])
                        except:
                            recog_decode_list.extend([None])

                    

                    if len(recog_decode_list) != input_roi_boxes.shape[0]:
                        return txt_result, im_result
                    
                    input_roi_boxes = input_roi_boxes[:, :8].reshape((-1, 4, 2))

                    for i, box in enumerate(input_roi_boxes):
                        #if np.amin(score) > 0:
                        if True: 

                            box = box[[0,3,2,1]]
                            #box = box.astype(np.int32)

                            box = box / [ratio_w, ratio_h]
                            box = box.astype(np.int32)

                            recognition_result = ground_truth_to_word(recog_decode_list[i])
                            txt_result.append(recognition_result)         
                            # Draw bounding box
                            im_result = cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                                
                else:
                    txt_result = []
                    im_result = im
        
    #tf.reset_default_graph() 
            return_dic['im_result_mtwi'] = im_result
            return_dic['txt_result_mtwi'] = txt_result       
            #return im_result, txt_result

if __name__ == '__main__':
    im_result, txt_result = predictmtwi('/home/wsw/workplace/deeplearning/DRWord/algorithm/test/img_55.jpg')
    print(txt_result)
    print(type(im_result))
    cv2.imwrite('/home/wsw/workplace/deeplearning/DRWord/algorithm/tset2.jpg', im_result)