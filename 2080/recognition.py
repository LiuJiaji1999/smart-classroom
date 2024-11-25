#-*- coding: utf-8 -*-
import math
import numpy as np
import os
import time
def get_angle(keypoint, x, y, z):
        angle = -1
        if keypoint[x,0] > 0 and keypoint[y,0] > 0 and keypoint[z,0] > 0:
            a = math.pow(keypoint[x,1]-keypoint[y,1],2) + math.pow(keypoint[x,0]-keypoint[y,0],2)
            b = math.pow(keypoint[z,1]-keypoint[y,1],2) + math.pow(keypoint[z,0]-keypoint[y,0],2)
            c = math.pow(keypoint[z,1]-keypoint[x,1],2) + math.pow(keypoint[z,0]-keypoint[x,0],2)
            if a > 0 and b > 0 and c > 0 and (a+b-c)/(2*math.sqrt(a*b)) >= -1 and (a+b-c)/(2*math.sqrt(a*b)) <= 1:
                angle = math.acos((a+b-c)/(2*math.sqrt(a*b)))
                angle = angle*180/math.pi
            #print(angle)
        return int(angle)

'''
Input:
    poseKeypoint: skeleton output by OpenPose
Output:
    0:other   1:headUp  2:headDown
'''
def HeadMetric(poseKeypoint):
    if poseKeypoint.ndim < 2:
        return 0
    else:

        angle = get_angle(poseKeypoint, 0, 1, 8)
        if angle > 120:
            return 1
        # elif angle < 40:
        #     return 2
        # else:
        #     return 0
        else:
            return 2


# 测试统计
'''
start = time.time()
all_poseKeypoint = []
path_list = os.listdir('/data/liujiaji/keda/')
for file in path_list:
    poseKeypoint = np.load("/data/liujiaji/keda/%s.npy" % (os.path.splitext(file)[0]))


    # 计算角度，得到抬头 低头指标
    result = []
    for i in range(poseKeypoint.shape[0]):
        head_num = HeadMetric(poseKeypoint[i])
        result.append(head_num)

    #统计出现的元素有哪些
    unique_data = np.unique(result)

    #统计某个元素出现的次数
    resdata = []
    for ii in unique_data:
        resdata.append(result.count(ii))
        if ii == 1:
            head = '抬头'
        elif ii == 2:
            head = '低头'
        else:
            head = '其他'
        print(os.path.splitext(file)[0],'中',head,'出现了',resdata[ii],'次')
end = time.time()
print("程序运行了%.5s秒:" % (end-start))
'''