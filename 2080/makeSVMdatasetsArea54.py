import json
from PIL import Image
import re
import numpy as np
import math
import sys
sys.path.append('/data/liujiaji/action/smartcam')
from linkeaction import SinglePersonSVM
from sklearn.metrics import classification_report,confusion_matrix
import recognition
import os
model = SinglePersonSVM(weights_path="/data/liujiaji/action/kedaclassroom/kedasvmpkl/kedasvm_weights_5107ppx.pkl")

# 判断点是否在多边形内
def isInterArea(testPoint,AreaPoint):#testPoint为待测点[x,y]
    LBPoint = AreaPoint[0]#AreaPoint为按顺时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    LTPoint = AreaPoint[1]
    RTPoint = AreaPoint[2]
    RBPoint = AreaPoint[3]
    a = (LTPoint[0]-LBPoint[0])*(testPoint[1]-LBPoint[1])-(LTPoint[1]-LBPoint[1])*(testPoint[0]-LBPoint[0])
    b = (RTPoint[0]-LTPoint[0])*(testPoint[1]-LTPoint[1])-(RTPoint[1]-LTPoint[1])*(testPoint[0]-LTPoint[0])
    c = (RBPoint[0]-RTPoint[0])*(testPoint[1]-RTPoint[1])-(RBPoint[1]-RTPoint[1])*(testPoint[0]-RTPoint[0])
    d = (LBPoint[0]-RBPoint[0])*(testPoint[1]-RBPoint[1])-(LBPoint[1]-RBPoint[1])*(testPoint[0]-RBPoint[0])
    if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
        return True
    else:
        return False

# 肢体连线 ，定义欧氏距离  [d1-d5]
def get_joint_distance(keypoint0,keypoint1):
    joint_distance = np.sqrt(sum(np.power((np.array(keypoint0) - np.array(keypoint1)), 2)))
    return joint_distance

# 关键点角度 ，定义肢体连线与正北方向的夹角 [5个角度]
def get_angle_north(keypoint0,keypoint1,dist): # keypoint0[x,y] > keypoint1[x,y]
    horizontal = keypoint0[0]-keypoint1[0]
    vertical = keypoint0[1]-keypoint1[1]
    angle_north = math.acos((math.pow(vertical,2)+math.pow(dist,2)-math.pow(horizontal,2))/(2*vertical*dist))
    if math.isnan(angle_north*180/math.pi):
        angle_north = 0
    else:
        angle_north = int(angle_north*180/math.pi)
    # print(angle_north)
    return angle_north


head_datasets = []
head_actionLabels = []
datasets = []
actionLabels = []


dict = json.load(open('/data/liujiaji/action/kedaclassroom/testKedaClass/5104_1-export.json','r',encoding='utf-8'))
# dict = json.load(open('/data/liujiaji/action/kedaAlldata/test/5104_2/5104_2-export.json','r',encoding='utf-8'))
print()
for key in dict['assets']:
    filetp = dict['assets'][key]
    ps = []
    cps = []
    ts = []
    ids = []

    if len(filetp['regions']) != 0:

        files = filetp['asset']['name']
        filename=os.path.splitext(files)[0]

        # pose = np.load('/data/liujiaji/action/kedaAlldata/test/5104_2/npy/%s.npy' %filename)
        pose = np.load('/data/liujiaji/action/kedaclassroom/testKedaClass/5104_1/npy/%s.npy' %filename)
        pose = pose[:,:9,:2]
        testpoint = []
        ori_allpose = []
        for i in range(pose.shape[0]):  #（44，25，3）44个人，25个关键点，3维坐标信息x,y,confidence
            x1, y1 = pose[i][1][:2]  #脖子点
            x8, y8 = pose[i][8][:2]  #骨盆点
            xc, yc = (x1+x8)*0.5, (y1+y8)*0.5 #分辨是否在 前排 中心
            testpoint.append([xc,yc]) # 该 key 图片下的 多个人的骨架点

            ori_allpose.append(pose[i])

       

        for i in range(len(filetp['regions'])):
            id = filetp['regions'][i]['id']  #学生id
            ids.append(id)

            tags =  filetp['regions'][i]['tags']   #多个标签
            ts.append(tags)

            pointxy = []
            # centerpoint = []
            point = filetp['regions'][i]['points']  # 该 key 图片下的目标框
            for j in range(len(point)):
                pointxy.append([point[j]['x'],point[j]['y']]) # 该 key 图片下的矩形目标框
            centerp = [(pointxy[0][0]+pointxy[1][0])/2,(pointxy[0][1]+pointxy[2][1])/2]
            # print(centerp)
            cps.append(centerp)
            ps.append(pointxy) # 该 key 图片下的 多个目标框
    else:
        continue

#5104    front[[0,619],[4,320],[1914,596],[1915,1080]]
#        center[[959,292],[1427,327],[741,828],[19,617]]
# 3A102   front[[0,735],[499,558],[1843,711],[1875,1080]]
#         center[[1089,483],[1358,505],[756,1008],[343,866]]
# 3C102   front[[11,1074],[5,648],[1920,487],[1920,798]]
        # center[[589,404],[1225,377],[1920,614],[1087,1061]]
#  5107    front[[0,412],[7,704],[1920,1080],[1920,625]]
            # center[[731,342],[1166,367],[751,866],[11,707]]

    for m in range(len(ps)):
        mind = 40
        index = -1
        # print(ps[m])
        for n in range(len(testpoint)):
            # print(testpoint[n])
            flag = isInterArea(testpoint[n],ps[m])  # 骨架点在目标框内
            flag_front = isInterArea(testpoint[n],[[0,619],[4,320],[1914,596],[1915,1080]]) #骨盆点在前排
            flag_centert = isInterArea(testpoint[n],[[959,292],[1427,327],[741,828],[19,617]]) #骨盆点在前排
            if flag :
                vec1 = np.array(cps[m]) #的中心点坐标
                vec2 = np.array(testpoint[n]) #骨骼点
                # #计算骨骼点与边框中点欧氏距离
                distance = np.sqrt(np.sum(np.square(vec1 - vec2)))  #二者的欧氏距离
                
                if distance < mind:
                    mind = distance
                    index = n
        if index != -1 :
            for o in range(len(ts[m])):
                # print(allpose[n][:8,:2])  # 只要 前8个点的xy坐标

                # 6分类
                if ts[m][o] != 'headDown' and ts[m][o] != 'headUp' and ts[m][o] != '' and  ts[m][o] != 'daze':
                    datasets.append(ori_allpose[index])
                    actionLabels.append(ts[m][o])
                else:
                    if ts[m][o] == 'headDown' or ts[m][o] == 'headUp':
                        head_datasets.append(ori_allpose[index])
                        head_actionLabels.append(ts[m][o])


# 测试抬头低头
print(len(head_datasets))
print(len(head_actionLabels))
test_label= []
for i in range(len(head_datasets)):
    headUpDown = recognition.HeadMetric(head_datasets[i])
    if headUpDown == 1:
        test_label.append('headUp')
    # elif headUpDown == 2:
    #     test_label.append('headDown')
    else:
        test_label.append('headDown')
# print(len(test_label))
print(classification_report(head_actionLabels, test_label))

print(len(datasets))
print(len(actionLabels))
# print(actionLabels)
datasets_list = []
for i in range(len(datasets)):

    xx = []
    yy = []
    for j in range(datasets[i].shape[0]):
        # datasets[i][j][0] = datasets[i][j][0]/1920
        # datasets[i][j][1] = datasets[i][j][1]/1080
        xx.append(datasets[i][j][0])
        yy.append(datasets[i][j][1])
    for j in range(datasets[i].shape[0]):
        datasets[i][j][0] = 960 * ((datasets[i][j][0] - min(xx)) / (max(xx) - min(xx)))
        datasets[i][j][1] = 540 * ((datasets[i][j][1] - min(yy)) / (max(yy) - min(yy)))



    dist10 = get_joint_distance( datasets[i][1], datasets[i][0])  #鼻子-脖子
    dist32 = get_joint_distance( datasets[i][3], datasets[i][2]) #右肩膀-右手肘
    dist43 = get_joint_distance( datasets[i][4], datasets[i][3]) #右手肘-右手腕
    dist65 = get_joint_distance( datasets[i][6], datasets[i][5]) #左肩膀-左手肘
    dist76 = get_joint_distance( datasets[i][7], datasets[i][6])  #左手肘-左手腕

    angle10 = get_angle_north( datasets[i][1], datasets[i][0],dist10) #鼻子-脖子 相对垂直90°方向的偏移  <90°
    angle32 = 90 + get_angle_north( datasets[i][3], datasets[i][2],dist32)#右肩膀-右手肘延申  相对水平180°方向的偏移 >90°
    angle43 = get_angle_north( datasets[i][4], datasets[i][3],dist43)#右手肘-右手腕 相对垂直90°方向的偏移  <90°
    angle65 = 90 + get_angle_north( datasets[i][6], datasets[i][5],dist65)#左肩膀-左手肘的延申 相对水平0°方向的偏移 >90°
    angle76 = get_angle_north( datasets[i][7], datasets[i][6],dist76) #鼻子-脖子 相对垂直90°方向的偏移  <90°

    # action_feature = [dist10,dist32,dist43,dist65,dist76]
    action_feature = [dist10,dist32,dist43,dist65,dist76,angle10,angle32,angle43,angle65,angle76]

    datalist = datasets[i].reshape(18).tolist()

    for k in range(len(action_feature)):
        datalist.append(action_feature[k])

    datasets_list.append(datalist)



pre_test = []
for i in range(len(datasets_list)):
    p_test = model.predict([datasets_list[i]])
    pre_test.append(p_test)
# print(pre_test)
print(classification_report(actionLabels, pre_test))
print(confusion_matrix(actionLabels, pre_test))




