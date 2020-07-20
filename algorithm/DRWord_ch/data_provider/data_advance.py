import numpy as np
from PIL import Image, ImageDraw  #第三方图像处理库
import os
import random
from tqdm import tqdm
import cv2

def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


def reorder_vertexes(xy_list):
    epsilon = 1e-4
    #xy_list.shape = 4*2
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    #np.argsort按列进行排序，返回相应下标
    ordered = np.argsort(xy_list, axis=0)
    #选择最小x,y的下标
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                    / (xy_list[index, 0] - xy_list[first_v, 0] + epsilon)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def resize_image(im, max_img_size=512):
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


def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False

def point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge):
    nth = -1
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth

def shrink(xy_list, ratio=0.2):
    epsilon = 1e-4
    #shrink_ratio = 0.2，shrink_side_ratio = 0.6
    #如果为0，返回原坐标
    if ratio == 0.0:
        return xy_list, xy_list

    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))
    # determine which are long or short edges
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
    short_edge = 1 - long_edge
    # cal r length array
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
    # cal theta array
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += epsilon
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
    # shrink two long edges
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)
    # shrink two short edges
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)
    return temp_new_xy_list, new_xy_list, long_edge

def shrink_edge(xy_list, new_xy_list, edge, r, theta, ratio=0.2):
    if ratio == 0.0:
        return
    start_point = edge
    end_point = (edge + 1) % 4
    long_start_sign_x = np.sign(
        xy_list[end_point, 0] - xy_list[start_point, 0])
    new_xy_list[start_point, 0] = \
        xy_list[start_point, 0] + \
        long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])
    long_start_sign_y = np.sign(
        xy_list[end_point, 1] - xy_list[start_point, 1])
    new_xy_list[start_point, 1] = \
        xy_list[start_point, 1] + \
        long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])
    # long edge one, end point
    long_end_sign_x = -1 * long_start_sign_x
    new_xy_list[end_point, 0] = \
        xy_list[end_point, 0] + \
        long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])
    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = \
        xy_list[end_point, 1] + \
        long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])

def generate_box(im_size, polys, tags, image_path, image):
    pixel_size = 4
    height, width = im_size
    #poly_mask = np.zeros((height, width), dtype=np.uint8)
    #score_map = np.zeros((height, width), dtype=np.uint8)
    #geo_map = np.zeros((height, width, 5), dtype=np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((height, width), dtype=np.uint8)
    gt = np.zeros((height // pixel_size, width // pixel_size, 7))

    image_name = image_path.split('/')[-1]
    #im = Image.open(image_path)
    im = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)

    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        xy_list = poly_tag[0]
        #tag = poly_tag[1]

        #if tag:
            #cv2.fillPoly(training_mask, xy_list.astype(np.int32)[np.newaxis, :, :], 0)

        #shrink_ratio = 0.2，长宽均按照0.2的比例进行缩放
        _, shrink_xy_list, _ = shrink(xy_list, 0.2)
        #shrink_side_ratio = 0.6 pixels between 0.2 and 0.6 are side pixels
        #长边按0.6的比例缩放，短边不变
        shrink_1, _, long_edge = shrink(xy_list, 0.6)


        #得分图边界？？？
        p_min = np.amin(shrink_xy_list, axis=0)
        p_max = np.amax(shrink_xy_list, axis=0)
        # floor of the float
        ji_min = (p_min / pixel_size - 0.5).astype(int) - 1
        # +1 for ceil of the float and +1 for include the end
        ji_max = (p_max / pixel_size - 0.5).astype(int) + 3
        #i为高度范围
        imin = np.maximum(0, ji_min[1])
        imax = np.minimum(height // pixel_size, ji_max[1])
        #j为宽度范围
        jmin = np.maximum(0, ji_min[0])
        jmax = np.minimum(width // pixel_size, ji_max[0])
        #检查所有像素
        for i in range(imin, imax):
            for j in range(jmin, jmax):
                #恢复到原图片大小？？？？
                px = (j + 0.5) * pixel_size
                py = (i + 0.5) * pixel_size
                #判断像素是否在文本框内
                if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):
                    #像素在文本框内
                    gt[i, j, 0] = 1
                    line_width, line_color = 1, 'red'
                    #是否为边界像素
                    ith = point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge)
                    vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                    if ith in range(2):
                        #是边界像素
                        gt[i, j, 1] = 1
                        if ith == 0:
                            line_width, line_color = 2, 'yellow'
                        else:
                            line_width, line_color = 2, 'green'
                        #是边界像素的头还是尾
                        gt[i, j, 2:3] = ith
                        #左上或右上(x,y)坐标——取决于是头还是尾
                        gt[i, j, 3:5] = xy_list[vs[long_edge][ith][0]] - [px, py]
                        #左下或右下(x,y)坐标——取决于是头还是尾
                        gt[i, j, 5:] = xy_list[vs[long_edge][ith][1]] - [px, py]

                #print(gt[i, j, :])
                '''    
                    draw.line([(px - 0.5 * pixel_size,
                                    py - 0.5 * pixel_size),
                                   (px + 0.5 * pixel_size,
                                    py - 0.5 * pixel_size),
                                   (px + 0.5 * pixel_size,
                                    py + 0.5 * pixel_size),
                                   (px - 0.5 * pixel_size,
                                    py + 0.5 * pixel_size),
                                   (px - 0.5 * pixel_size,
                                    py - 0.5 * pixel_size)],
                                   width=line_width, fill=line_color) 
    im.save(os.path.join('/home/wsw/deeplearning/advancedFOTS/test/', image_name))   
    '''
    #print(gt[40:50,40:50, 0:7])
    return gt
