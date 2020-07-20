# coding=utf-8
import numpy as np


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    #当集合中存在相同的值时，则返回false
    return not region.isdisjoint(neighbor)


def region_neighbor(region_set):
    '''
    region_list[m]：{(i,j), (k,h)} 像素的下标集合
    '''
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor

#region_list=[ {(i,j),(k,h)}, {(l,n)} ]，同一集合内的像素认为属于同一个
def region_group(region_list):
    #S指region_list中有多少个集合，S=[0,1,2,3.....,len(region_list)-1]
    S = [i for i in range(len(region_list))]
    
    D = []
    #集合个数大于0
    while len(S) > 0:
        #m为S中第一个元素
        m = S.pop(0)
        #是否只有一个集合
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    #D：([l,i,j], [m], .....)
    return D

def rec_region_merge(region_list, m, S):
    '''
    region_list：[ {(i,j),(k,h)}, {(l,n)} ]，同一集合内的像素认为属于同一个
    S：region_list中有多少个集合，S=[0,1,2,3.....,len(region_list)-1]
    m：m=S.pop(0) S中的第一个元素，0/1/2.....
    '''
    rows = [m]
    tmp = []
    #S中不包含m
    for n in S:
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))
    return rows

#side_vertex_pixel_threshold = 0.9
def nms(predict, activation_pixels, threshold=0.9):
    pixel_size = 4
    epsilon = 1e-4
    #去重后的得分图像素坐标
    region_list = []
    #activation_pixels[0]保存行坐标， activation_pixels[1]保存列坐标
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            #有相同值时进行合并
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})
    
    #region_list=[ {(i,j),(k,h)}, {(l,n)}, ....... ]
    #D：([l,i,j], [m], .....)，其中i,j,l,m指代region_list中元素下标
    #D中每个元素代表一个文本行？？？
    D = region_group(region_list)

    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))

    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            #ij指代region_list中元素，得分图像素坐标
            for ij in region_list[row]:
                #print(ij)
                #是否为边界元素的得分
                score = predict[ij[0], ij[1], 1]
                #print('score shape is : ', score)
                if score >= threshold:
                    #头、尾元素得分
                    ith_score = predict[ij[0], ij[1], 2:3]
                    trunc_threshold=0.1
                    if not (trunc_threshold <= ith_score < 1 - trunc_threshold):
                        #np.around四舍五入 0.9<ith_score or ith_score<0.1
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * pixel_size
                        py = (ij[0] + 0.5) * pixel_size
                        #p_v=(2,2)
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7],
                                              (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v

        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + epsilon)
        #print('total_score : ', score_list[g_th])
        #print('quad_list : ', quad_list[g_th])
    return score_list, quad_list
