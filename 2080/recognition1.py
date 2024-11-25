#-*- coding: utf-8 -*-
import sys
import cv2
import os
from sys import platform
import numpy as np
import time
# from skeleton1 import OpenPose
from skeleton2 import OpenPose
import math
import json
import yaml
import datetime
from multiprocessing import Process,Queue
#from util import *
#import util


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

def draw(img, result):
    font = cv2.FONT_HERSHEY_COMPLEX
    for i in range(0, result.poseKeypoints.shape[0]):
        angle = get_angle(result.poseKeypoints[i], 0, 1, 8)
        cv2.putText(img,str(angle),(int(result.poseKeypoints[i,1,0]),int(result.poseKeypoints[i,1,1])),font,1, (0,252,142), 2)
    cv2.imwrite('angle1.png',img)
    print('finish')

'''
Input:
    result: skeleton output by OpenPose
    ClassNumber: course number
Output:
    AttendanceRate
'''
def AttendanceRate(result, ClassNumber):
    #with open('/raid/workspace/openpose/se/model/skeleton/StudentNumber.json','r') as load_f:
    #    doc = json.load(load_f)

    if result.getPoseKeypoints().ndim < 3:
        return 0
    else:
        return result.poseKeypoints.shape[0]#/doc[ClassNumber]
# original x*y:1920*1080
# resize x*y:640*360
#"frontArea":
# resize:[{"x": 3, "y": 135}, {"x": 635, "y": 213}, {"x": 631, "y": 352}, {"x": 1, "y": 245}]
# original:[[9,405],[1905,639],[1893,1056],[3,735]]
def FrontRate(result, FrontArea=np.array([[5,598],[1527,1061],[1716,430],[468,328]])):
    #print(FrontArea)
    if FrontArea is None:
        return 0
    if result.getPoseKeypoints().ndim > 1 and FrontArea.size > 0:
        FrontNumber = 0
        for i in range(0, result.poseKeypoints.shape[0]):
            count = 0
            for j in range(0,8):
                if cv2.pointPolygonTest(FrontArea, (result.poseKeypoints[i][j,0],result.poseKeypoints[i][j,1]), False) > 0:
                    count = count + 1
            if count > 3:
                FrontNumber = FrontNumber + 1
        return FrontNumber
    else:
        return 0

#"centerArea": [{"x": 85, "y": 90}, {"x": 628, "y": 138}, {"x": 638, "y": 203}, {"x": 6, "y": 130}]
# resize:[[85, 90], [628, 138], [638, 203], [6, 130]]
# original:[[255, 270], [1884, 414], [1914, 609], [18, 390]]
def CenterRate(result, MiddleArea=np.array([[483, 325], [1887, 441], [1908, 208], [870, 188]])):
    if MiddleArea is None:
        return 0
    if result.getPoseKeypoints().ndim > 1 and MiddleArea.size > 0:
        MiddleNumber = 0
        for i in range(0, result.poseKeypoints.shape[0]):
            count = 0
            for j in range(0,8):
                if cv2.pointPolygonTest(MiddleArea, (result.poseKeypoints[i][j,0],result.poseKeypoints[i][j,1]), False) > 0:
                    count = count + 1
            if count > 2:
                MiddleNumber = MiddleNumber + 1
        return MiddleNumber
    else:
        return 0


'''
Input:
    result: skeleton output by OpenPose
    cnf_file: threshold configure file
Output:
    UpRate, DownRate, LieRate
'''
# def HeadMetric(result, cnf_file="/datapool/workspace/VideoAnalysis/SkeletonModel/threshold.yml"):
# up_down: 120
# down_lie: 40
def HeadMetric(result, cnf_file="/data/workspace/yaojiapeng/yaojiapeng/skeletonmodel-develop/threshold.yml"):
    '''
    UpNumber = 0
    DownnNumber = 0
    LieNumber = 0
    y = yaml.load(open(cnf_file,'r'))

    for i in range(0, result.poseKeypoints.shape[0]):
        angle = get_angle(result.poseKeypoints[i], 0, 1, 8)
        if angle > y['up_down']:
            UpNumber = UpNumber + 1
        else:
            LieNumber = LieNumber + 1
    '''
    standNumber = 0
    UpNumber = 0
    DownnNumber = 0
    LieNumber = 0
    action_list = [' ','stand', 'raise', 'lie']
    index = 0
    y = yaml.load(open(cnf_file,'r'),Loader=yaml.FullLoader)
    font = cv2.FONT_HERSHEY_COMPLEX

    if result.getPoseKeypoints().ndim < 3:
        return 0,0,0
    else:
        for i in range(0, result.poseKeypoints.shape[0]):
            angle = get_angle(result.poseKeypoints[i], 17, 1, 8)
            angle1 = get_angle(result.poseKeypoints[i], 9,10,11)
            #print('1')
            angle2 = get_angle(result.poseKeypoints[i], 12,13,14)
            if (angle1 > 0 and angle1 > 160) and (angle2 > 0 and angle2 > 160):
                if result.poseKeypoints[i][1,0] > 0:
                    index = 1
                    standNumber = standNumber+1
            elif angle > y['up_down']:
                UpNumber = UpNumber + 1
                index = 2
            #elif angle > y['down_lie']:
            #    DownnNumber = DownnNumber + 1
            else:
                LieNumber = LieNumber + 1
                index = 3
            # elif angle < y['down_lie']:
            #     LieNumber = LieNumber + 1
            #     index = 3
            # #elif angle > y['down_lie']:
            # #    DownnNumber = DownnNumber + 1
            # else:
            #     UpNumber = UpNumber + 1
            #     index = 2
        #return UpNumber/result.poseKeypoints.shape[0], DownnNumber/result.poseKeypoints.shape[0], LieNumber/result.poseKeypoints.shape[0]
        return standNumber, UpNumber, LieNumber#, DownnNumber/result.poseKeypoints.shape[0], LieNumber/result.poseKeypoints.shape[0]



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    opper = OpenPose()
    '''
    ###
    img = cv2.imread('/data/liuzhenyu/results/c1_front_11010.png')
    print("OpenPose 测试 ...")
    time_start = time.time()
    result = opper.infer(img)
    attend=AttendanceRate(result, 'class0001')
    stand,up,lie=HeadMetric(result)
    front=FrontRate(result)
    center=CenterRate(result)
    print('到课:'+ str(attend)+';前排:'+ str(front)+';中心:'+ str(center))
    print('走动:'+ str(stand)+';抬头:'+ str(up)+';低头:'+ str(lie))
    time_end = time.time()
    print('OpenPose 测试完成，耗时：',time_end - time_start,'秒.')

    ###
    '''

    for root, dirs,files in os.walk("/data/liujiaji/kedatest"):
        for file in files:
            filepath=os.path.join(root,file)
            img = cv2.imread(filepath)
            print("OpenPose 测试 ...")
            time_start = time.time()
            result = opper.infer(img)
            print(file)
            attend=AttendanceRate(result, 'lesson5_18')
            stand,up,lie=HeadMetric(result)
            front=FrontRate(result)
            center=CenterRate(result)
            print('到课:'+ str(attend)+';前排:'+ str(front)+';中心:'+ str(center))
            print('走动:'+ str(stand)+';抬头:'+ str(up)+';低头:'+ str(lie))
    # while True:
    #     try:
    #         result = opper.infer(img)
    #         print('出席人数:'+ str(AttendanceRate(result, 'class0001')))
    #         print('抬头，低头，趴桌人数:'+str(HeadMetric(result)))
    #         count = count + 1
    #         print(count)
    #     except Exception as e:

    #         print( datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #         print(e)
            time_end = time.time()
            print('OpenPose 测试完成，耗时：',time_end - time_start,'秒.')
            print_log=open("./keda/result_lesson4444.txt",'a')
            print(file,file=print_log)
            print('到课:'+ str(attend)+';前排:'+ str(front)+';中心:'+ str(center),file=print_log)
            print('走动:'+ str(stand)+';抬头:'+ str(up)+';低头:'+ str(lie),file=print_log)
            print('用时:'+str(time_end - time_start)+'秒',file=print_log)




