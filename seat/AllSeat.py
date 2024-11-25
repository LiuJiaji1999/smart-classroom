import re
import numpy as np
from scipy import optimize
import json
import cv2



# 1 划分排排的座位信息          
def getSeat(filepath):
    with open(filepath, 'r') as obj:
        dict = json.load(obj)
    areas = []
    for i in range(len(dict['cameraConfig'])):
        seatConfig = dict['cameraConfig'][i]['seatConfig']
        for j in range(len(seatConfig)):
            area = seatConfig[j]['area']
            areas.append(area)
    box = []
    for k in range(len(areas)):
        point = [[areas[k][0]['x'],areas[k][0]['y']],[areas[k][1]['x'],areas[k][1]['y']],[areas[k][2]['x'],areas[k][2]['y']],[areas[k][3]['x'],areas[k][3]['y']]]
        box.append(point)
    pts = np.array(box)
    return box

# 2 在排排座位上，切分各个座位
def aloneSeat(box): 
    line1 = [box[0], box[1]]
    dis1_x = line1[1][0] - line1[0][0]
    dis1_y = line1[1][1] - line1[0][1]
    seat_dis1_x = dis1_x/4
    seat_dis1_y = dis1_y/4
    relt1 = [box[0]]  # [[160, 719]]
    x,y = line1[0][0],line1[0][1]
    for i in range(4):
        x += seat_dis1_x
        y += seat_dis1_y
        x = int(x)
        y = int(y)
        new_seat = [x,y]
        relt1.append(new_seat)
    line2 = [box[3], box[2]]
    dis2_x = line2[1][0] - line2[0][0]
    dis2_y = line2[1][1] - line2[0][1]
    seat_dis2_x = dis2_x/4
    seat_dis2_y = dis2_y/4
    relt2 = [box[3]]
    x,y = line2[0][0],line2[0][1]
    for i in range(4):
        x += seat_dis2_x
        y += seat_dis2_y
        x = int(x)
        y = int(y)
        line = [x,y]
        relt2.append(line)
  
    ss = [] #每一排所有座位
    ss1 = [relt1[0],relt1[1],relt2[1],relt2[0]]
    ss2 = [relt1[1],relt1[2],relt2[2],relt2[1]]
    ss3 = [relt1[2],relt1[3],relt2[3],relt2[2]]
    ss4 = [relt1[3],relt1[4],relt2[4],relt2[3]]
    ss.append(ss1)
    ss.append(ss2)
    ss.append(ss3)
    ss.append(ss4)  
    return ss

# 3 先画映射的骨骼点
def yinshe_skeleton(skeleton_result):
    relation = find_neighbor_skeleton(skeleton_result)
    failed = []
    for relate in relation:
        target_id, neighb1, neighb2 = relate
        n1_x0, n1_y0 = skeleton_result[neighb1][0][:2] # 第一个邻近人的骨骼点的x0，x1,x8
        n1_x1, n1_y1 = skeleton_result[neighb1][1][:2]
        n1_x8, n1_y8 = skeleton_result[neighb1][8][:2]
        n1_xc, n1_yc = (n1_x1+n1_x8)/2, (n1_y1+n1_y8)/2

        n2_x0, n2_y0 = skeleton_result[neighb2][0][:2]
        n2_x1, n2_y1 = skeleton_result[neighb2][1][:2]
        n2_x8, n2_y8 = skeleton_result[neighb2][8][:2]
        n2_xc, n2_yc = (n2_x1+n2_x8)/2, (n2_y1+n2_y8)/2

        vector1 = [n1_xc-n1_x0, n1_yc-n1_y0] #第一个邻近人的 中心点---鼻子的距离
        vector2 = [n2_xc-n2_x0, n2_yc-n2_y0]
        vectorc = [(vector1[0]+vector2[0])/2, (vector1[1]+vector2[1])/2]
        
        x0, y0 = skeleton_result[target_id][0][:2]
        xc, yc = x0+vectorc[0], y0+vectorc[1]
        failed.append([target_id,xc,yc])

    return failed

# 4 在映射好的骨骼点上，补全其他本身找到的点，进而得到所有的骨骼点图
def findSite(skeleton_result,seat_result):
    failed = yinshe_skeleton(skeleton_result) #得到映射后的xc.yc坐标，进而补充所有的xc,yc，进行人座匹配
    match = []
    for j in range(len(seat_result)): #每排4个座位
        for i in range(skeleton_result.shape[0]):  #（44，25，3）44个人，25个关键点，3维坐标信息x,y,confidence
            x1, y1 = skeleton_result[i][1][:2]  #脖子点
            x6, y6 = skeleton_result[i][6][:2]  #左手肘点
            x8, y8 = skeleton_result[i][8][:2]  #盆骨点 
                
            if x1==0 or x8==0:  #为0是因为有些人的骨架信息没检测到 
                continue
            xc, yc = (x1+x8)*0.5, (y1+y8)*0.5
            if isInterArea([xc,yc], seat_result[j]):  #判断点是否在多边形区域内！
                match.append((i,p,j))  #直接输出第 i 个骨架在第 j 座位
        for k in range(len(failed)):
            if isInterArea([failed[k][1],failed[k][2]], seat_result[j]):  
                match.append((failed[k][0],p,j))    
    return match



# 找邻近骨骼信息
def find_neighbor_skeleton(skeleton_result):
    result = []
    for i in range(skeleton_result.shape[0]): #44
        x0, y0 = skeleton_result[i][0][:2] #鼻子
        x8, y8 = skeleton_result[i][8][:2]  # 盆骨
        x1, y1 = skeleton_result[i][1][:2]  #脖子
        xc, yc = (x1+x8)*0.5, (y1+y8)*0.5  #中心点
        min_distance = float('inf') # 正无穷  最小距离
        submin_distance = float('inf')
        neighbor = [i, 5000, 5000] # 找到的第一个人的id target_id, neighb1, neighb2 
        if x0*x1*x8 != 0:
            continue
        for j in range(skeleton_result.shape[0]): #44
            if i == j: #表示是同一个人
                continue
            temp_x0, temp_y0 = skeleton_result[j][0][:2]
            temp_x1, temp_y1 = skeleton_result[j][1][:2]
            temp_x8, temp_y8 = skeleton_result[j][8][:2]
            if temp_x0*temp_x1*temp_x8 == 0:
                continue
            current_distance = (x0-temp_x0)**2+(y0-temp_y0)**2 
            if current_distance < min_distance:
                min_distance = current_distance
                neighbor[1] = j #
            elif current_distance < submin_distance:
                submin_distance = current_distance
                neighbor[2] = j
        result.append(neighbor)
    return result

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


allbox = getSeat('javaSeatV2_houpai.json')
print(allbox)
print(len(allbox))
tt = []
for i in range(len(allbox)):
    sseat = aloneSeat(allbox[i])
    tt.append(sseat)
m = []
for p in range(len(tt)): # i是第几排
    # print(tt[p])
    skeleton_result = np.load('houpai.npy')  #由骨架结果计算骨架中心点
    match = findSite(skeleton_result,tt[p]) #第几个座位
    m.append((match))
print(m)
