# 1. 每4个点拟合成一条线， 
# 2. 每条线自己做分割
# 每个点的四个坐标按照从左下开始逆时针记录
import re
import numpy as np
from scipy import optimize
import json
import cv2


class all(object):
    def __init__(self, json_path, skeleton_path): 
         self.json_path = json_path
         self.skeleton_path = skeleton_path

    def getSkeleton(self):
        skeleton_result = np.load(self.skeleton_path)  #由骨架结果计算骨架中心点
        return skeleton_result

    # 1 划分排排的座位信息          
    def getSeat(self):
        # with open('javaSeatV2.json', 'r') as obj:
        with open(self.json_path, 'r') as obj:
            dict = json.load(obj)
        img = cv2.imread('qianpai/frame_1500.jpg')
        # print(dict['cameraConfig'])
        # print(len(dict['cameraConfig']))
        areas = []
        for i in range(len(dict['cameraConfig'])):
            seatConfig = dict['cameraConfig'][i]['seatConfig']
            # print(seatConfig)
            for j in range(len(seatConfig)):
                area = seatConfig[j]['area']
                areas.append(area)
        seat = []
        for k in range(len(areas)):
            # print(areas[k])
            point = [[areas[k][0]['x'],areas[k][0]['y']],[areas[k][1]['x'],areas[k][1]['y']],[areas[k][2]['x'],areas[k][2]['y']],[areas[k][3]['x'],areas[k][3]['y']]]
            seat.append(point)
        # print(len(box))
        # print(box[0])
        pts = np.array(seat)
        # print(pts)
        cv2.polylines(img, pts, True, (0, 0, 255), 2)
        cv2.imwrite("dangxiao_test/1_paipai.png", img)
        return seat

    # 4 在排排座位上，切分各个座位
    def aloneSeat1(self,seat): 
        line1 = [seat[0], seat[1]]
        dis1_x = line1[1][0] - line1[0][0]
        dis1_y = line1[1][1] - line1[0][1]
        seat_dis1_x = dis1_x/4
        seat_dis1_y = dis1_y/4
        relt1 = [seat[0]]  # [[160, 719]]
        x,y = line1[0][0],line1[0][1]
        # print(relt1)
        for i in range(4):
            x += seat_dis1_x
            y += seat_dis1_y
            x = int(x)
            y = int(y)
            new_seat = [x,y]
            relt1.append(new_seat)

        line2 = [seat[3], seat[2]]
        dis2_x = line2[1][0] - line2[0][0]
        dis2_y = line2[1][1] - line2[0][1]
        seat_dis2_x = dis2_x/4
        seat_dis2_y = dis2_y/4

        relt2 = [seat[3]]
        x,y = line2[0][0],line2[0][1]
        for i in range(4):
            x += seat_dis2_x
            y += seat_dis2_y
            x = int(x)
            y = int(y)
            line = [x,y]
            relt2.append(line)

        # img = cv2.imread('dangxiao_test/1_paipai.png')
        img = cv2.imread('dangxiao_test/2_all_alone.png')
        cv2.line(img, tuple(relt1[1]), tuple(relt2[1]),(255,0,0),3)
        cv2.line(img, tuple(relt1[2]), tuple(relt2[2]),(255,0,0),3)
        cv2.line(img, tuple(relt1[3]), tuple(relt2[3]),(255,0,0),3)
        cv2.imwrite("dangxiao_test/2_all_alone.png", img)

        ss = [] #第一排所有座位
        ss1 = [relt1[0],relt1[1],relt2[1],relt2[0]]
        ss2 = [relt1[1],relt1[2],relt2[2],relt2[1]]
        ss3 = [relt1[2],relt1[3],relt2[3],relt2[2]]
        ss4 = [relt1[3],relt1[4],relt2[4],relt2[3]]
        ss.append(ss1)
        ss.append(ss2)
        ss.append(ss3)
        ss.append(ss4)  
        return ss

    # 2 已知所有座位，去切分所有各个座位
    def aloneSeat2(seat): 
        #先得到每排座位的线
        lineR = []
        # print('所有座位信息：',seat)
        for i in range(len(seat)):
            line1 = [seat[i][0],seat[i][1]]
            lineR.append(line1)
            # line2 = [seat[i][3],seat[i][2]]
            # lineR.append(line2)
        # print(lineR)
        lineR.append([seat[len(seat)-1][3],seat[len(seat)-1][2]])
        print(len(lineR))
        # print(np.array(lineR))
        # print(lineR[0])  # 第一条线 [[160, 719], [1148, 981]]
        # print(lineR[0][0]) # 第一条线的 xy坐标 [160, 719]
        # print(lineR[0][0][0]) # 第一条线的 x坐标 160
        # 计算每条线的 x y差值并 等分4
        ll = [] # 存放新生成的坐标点
        lineAll = []
        for i in range(len(lineR)): #0-9 共10条线
            # print('第1个坐标的x值：',lineR[i][0][0])
            # print('第1个坐标的y值：',lineR[i][0][1])
            # print('第2个坐标的x值：',lineR[i][1][0])
            # print('第2个坐标的y值：',lineR[i][1][1])
            
            dist_x = lineR[i][1][0] - lineR[i][0][0]
            dist_y = lineR[i][1][1] - lineR[i][0][1]
            seat_dist_x = int(dist_x/4)
            seat_dist_y = int(dist_y/4)
            # print('每条线的平均x间隔：',seat_dist_x)
            line = [lineR[i][0]] # 先初始化每排的第一个坐标，然后存放同一条线上新生成的坐标点
            x,y = lineR[i][0][0],lineR[i][0][1]
            for i in range(4):
                x += seat_dist_x
                y += seat_dist_y
                new_seat = [x,y] # 存放新生成的坐标点
                # print('新增的坐标点：',new_seat)
                ll.append(new_seat)
                line.append(new_seat)
            lineAll.append(line)
        # print('新生成的所有点',ll)
        # print(len(ll))
        # print('所有点：',lineAll)
        all = np.array(lineAll) 
        # print(all)
        # print(all.shape[1]) #(7,5,2) 7条线，5个坐标，每个坐标2个值xy
        a = np.reshape(all,(-1,2))
        a = a.tolist()
        ss = []

        for i in range(len(a)):
            if a[i] < a[i+1]:
                aloneSeat = [a[i],a[i+1],a[i+6],a[i+5]]   
                ss.append(aloneSeat) 
                if i == len(a)-7:
                    break
        # print(ss)
        ase = np.array(ss)
        # print(ase[0])
        # print(ase.shape) 

        img = cv2.imread('school/1_paipai_houpai.png')
        for i in range(len(ll)):
            cv2.line(img, tuple(ll[i]), tuple(ll[i+4]),(255,0,0),3)
            i += 1
            if i == len(ll)-4:
                break  
        cv2.imwrite("school/2_all_alone_houpai.png", img)
        return ss


    # 3 先画映射的骨骼点
    def yinshe_skeleton(self,skeleton_result):
        relation = self.find_neighbor_skeleton(skeleton_result)
        # print(relation)
        # print(len(relation))
        img = cv2.imread('dangxiao_test/2_all_alone.png')
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
            # print([xc,yc])
            failed.append([target_id,xc,yc])

            cv2.line(img, tuple([int(x0),int(y0)]), tuple([int(xc),int(yc)]), (0,0,225), 3)
            cv2.circle(img, tuple([int(xc),int(yc)]),2,(0,255,0), 4)
        cv2.imwrite('dangxiao_test/3_show_failed_result.png', img)
        return failed

    # 4 在映射好的骨骼点上，补全其他本身找到的点，进而得到所有的骨骼点图
    def findSite(self,skeleton_result,seat_result):
        failed = self.yinshe_skeleton(skeleton_result) #得到映射后的xc.yc坐标，进而补充所有的xc,yc，进行人座匹配
        match = []
        img = cv2.imread('dangxiao_test/3_show_failed_result.png')
        for j in range(len(seat_result)): #每排4个座位
            for i in range(skeleton_result.shape[0]):  #（44，25，3）44个人，25个关键点，3维坐标信息x,y,confidence
                x1, y1 = skeleton_result[i][1][:2]  #脖子点
                x6, y6 = skeleton_result[i][6][:2]  #左手肘点
                x8, y8 = skeleton_result[i][8][:2]  #盆骨点 
                    
                if x1==0 or x8==0:  #为0是因为有些人的骨架信息没检测到 
                    continue
                xc, yc = (x1+x8)*0.5, (y1+y8)*0.5
                # print(xc,yc)
                cv2.line(img, tuple([int(x1),int(y1)]), tuple([int(xc),int(yc)]), (0,0,225), 3)
                cv2.circle(img, tuple([int(xc),int(yc)]),2,(0,255,0), 4) 

                if self.isInterArea([xc,yc], seat_result[j]):  #判断点是否在多边形区域内！
                    match.append((i,p,j))  #直接输出第 i 个骨架在第 p 排第j列
            for k in range(len(failed)):
                if self.isInterArea([failed[k][1],failed[k][2]], seat_result[j]):  
                    match.append((failed[k][0],p,j))    
        cv2.imwrite('dangxiao_test/4_show_all_skeleton.png',img)
        return match



    # 找邻近骨骼信息
    def find_neighbor_skeleton(self,skeleton_result):
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
                # print(current_distance)
                if current_distance < min_distance:
                    min_distance = current_distance
                    neighbor[1] = j #
                elif current_distance < submin_distance:
                    submin_distance = current_distance
                    neighbor[2] = j
            result.append(neighbor)
        # print(result)
        return result

    # 判断点是否在多边形内
    def isInterArea(self,testPoint,AreaPoint):#testPoint为待测点[x,y]
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


# allbox = getSeat('javaSeatV2.json')
# tt = []
# for i in range(len(allbox)):
#     sseat = aloneSeat1(allbox[i])
#     tt.append(sseat)
# m = []
# for p in range(len(tt)): # i是第几排
#     skeleton_result = np.load('npy/frame_13422.npy')  #由骨架结果计算骨架中心点
#     match = findSite(skeleton_result,tt[p]) #第几个座位
#     print('骨骼id，排，列:',match)
#     m.append(match)
# # print('骨骼id，排，列:',m)



test = all('javaSeatV2.json','npy/frame_1500.npy')
allseat = test.getSeat()
print(allseat)
print(len(allseat))
tt = []
for i in range(len(allseat)):
    sseat = test.aloneSeat1(allseat[i])
    tt.append(sseat)
m = []
for p in range(len(tt)): # i是第几排
    # skeleton_result = np.load('npy/frame_13422.npy')  #由骨架结果计算骨架中心点
    skeleton_result = test.getSkeleton()
    match = test.findSite(skeleton_result,tt[p]) #第几个座位
    print('骨骼id，排，列:',match)
    m.append(match)
# print('骨骼id，排，列:',m)

tt = np.array(tt)
print('座位信息：',tt.shape)