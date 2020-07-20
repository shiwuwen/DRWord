import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import locality_aware_nms as nms_locality
#import lanms
from nms import nms
from bktree import BKTree, levenshtein, list_words

tf.app.flags.DEFINE_string('test_data_path', '/home/wsw/temp/MTWI/icpr_mtwi_task3/image_test/', '')  
#/home/wsw/temp/ICDAR2015/ch4_test_images/ /home/wsw/temp/test1/ /home/wsw/temp/MTWI/icpr_mtwi_task3/image_test/
tf.app.flags.DEFINE_string('gpu_list', '-1', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpointmtwi/', '') #pretrainedmodel/SynthText/
tf.app.flags.DEFINE_string('output_dir', 'mtwi_outputs/', '') #ch4_test_outputs_20200115/
tf.app.flags.DEFINE_bool('no_write_images', True, 'do not write images')
# tf.app.flags.DEFINE_bool('use_vacab', True, 'strong, normal or weak')

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

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    #print(files)
    return files


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

def resize_image2(im, max_img_size=512):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height

def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
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
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

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

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    """
    if FLAGS.use_vacab and os.path.exists("./vocab.txt"):
        bk_tree = BKTree(levenshtein, list_words('./vocab.txt'))
        # bk_tree = bktree.Tree()
    """            
    with tf.get_default_graph().as_default():
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

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            index = len(im_fn_list)
            for im_fn in im_fn_list:

                print('still have {} pictures left'.format(index))
                print(im_fn)

                #image for draw quad
                im = cv2.imread(im_fn)[:, :, ::-1]       
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                
                
                #method 2
                #im = image.load_img(im_fn)
                #im_draw = im.copy
                #d_wight, d_height = resize_image2(im)
                #im_resized = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
                #im_draw = cv2.resize(im_draw, (int(d_wight), int(d_height)))
                #ratio_w = d_wight / im.width
                #ratio_h = d_height / im.height

                img = image.img_to_array(im_resized)
                img = preprocess_input(img, mode='tf')               

                shared_feature_map, geometry = sess.run([shared_feature, f_geometry], feed_dict={input_images: [img]})
                
                geometry = np.squeeze(geometry, axis=0)
                print('geometry is : ', (np.array(geometry)).shape)
                geometry[:, :, :3] = sigmoid(geometry[:, :, :3])
                #print('geometry : ' ,geometry[40:50,40:50,0:7])

                pixel_threshold = 0.9
                cond = np.greater_equal(geometry[:, :, 0], pixel_threshold)
                activation_pixels = np.where(cond)
                print('activation_pixels is : ', (np.array(activation_pixels)).shape)

                #for i,j in zip(activation_pixels[0], activation_pixels[1]):
                    #print(geometry[i, j, 0:7])

                #boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                quad_scores, boxes = nms(geometry, activation_pixels)
                #print('boxes : ',boxes)

                input_roi_boxes = []
                im_txt = None
                #for ICDAR
                #res_file_path = os.path.join(FLAGS.output_dir, 'res_' + '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                #for MTWI
                res_file_path = os.path.join(FLAGS.output_dir, '{}.txt'.format(im_fn[:-4].split('/')[-1]))

                with open(res_file_path, 'w') as f:
                    for score, box in zip(quad_scores, boxes): 
                        if np.amin(score) > 0:
                            
                            #for ICDAR
                            #box = box[[0,3,2,1]]
                            #box = sort_poly(box.astype(np.int32))

                            box = box / [ratio_w, ratio_h]
                            box = box.astype(np.int32)
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                            im_txt = cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                
                #to show how many pictures left
                index -= 1
                print('------------------------------------')

                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    # cv2.imwrite(img_path, im[:, :, ::-1])
                    if im_txt is not None:
                        cv2.imwrite(img_path, im_txt)

if __name__ == '__main__':
    #在执行启动之前要解析以tensorflow方式定义的变量，
    #而tensorflow的变量定义有4种类型，整型，字符，浮点，布尔。
    tf.app.run()
