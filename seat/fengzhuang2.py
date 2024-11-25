
import numpy as np
import json
import cv2
from datetime import datetime

class ssMatch(object):
    def __init__(self, json_path, skeleton_path): 
         self.json_path = json_path
         self.skeleton_path = skeleton_path
         

    # 1 读取座位标注，划分排排的座位信息          
    def getSeat(self,img):
        img = img
        with open(self.json_path, 'r') as obj:
            dict = json.load(obj)
        users = []
        areas = []
        for j in range(len(dict['seatConfig'])):
            area = dict['seatConfig'][j]['seatListArea']
            areas.append(area)
            userId = dict['seatConfig'][j]['userIdList']
            users.append(userId)
        seat = []
        for k in range(len(areas)):
            point = [[areas[k][0]['x'],areas[k][0]['y']],[areas[k][1]['x'],areas[k][1]['y']],[areas[k][2]['x'],areas[k][2]['y']],[areas[k][3]['x'],areas[k][3]['y']]]
            seat.append(point)
        pts = np.array(seat)
        cv2.polylines(img, pts, True, (0, 128, 255), 2)
        # cv2.imwrite("dangxiao_test/1_paiSeat.png", img)
        return seat, users

    # 2 读取检测到的骨架信息
    def getSkeleton(self):
        skeleton_result = np.load(self.skeleton_path)  #由骨架结果计算骨架中心点
        return skeleton_result


    # 3 在排排座位上，切分各个座位
    def aloneSeat(self,seat): 
        line1 = [seat[0], seat[1]]
        dis1_x = line1[1][0] - line1[0][0]
        dis1_y = line1[1][1] - line1[0][1]
        seat_dis1_x = dis1_x/4
        seat_dis1_y = dis1_y/4
        relt1 = [seat[0]]  # [[160, 719]]
        x,y = line1[0][0],line1[0][1]
        for i in range(4):
            x += seat_dis1_x
            y += seat_dis1_y
            x = int(x)
            y = int(y)
            new_seat = [x, y]
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
            line = [x, y]
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

    #  4 映射未检测到的骨骼信息
    def yinshe_skeleton(self,skeleton_result):
        # img = cv2.imread('dangxiao_test/2_aloneSeat.png')
        relation = self.find_neighbor_skeleton(skeleton_result)
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
            
        #     cv2.circle(img, tuple([int(xc),int(yc)]),2,(0,255,0), 4)
        # cv2.imwrite('dangxiao_test/3_failedSkeleton.png', img)
        return failed

    

    # 5 在映射好的骨骼点上，补全其他本身找到的点，进而得到所有的骨骼点图

    ####此处有修改，新增返回值not_match_point&box，增加输入值img用于可视化
    def findSite(self,skeleton_result,seat_result,img):
        img = img
        failed = self.yinshe_skeleton(skeleton_result) #得到映射后的xc.yc坐标，进而补充所有的xc,yc，进行人座匹配
        match = []
        not_match_box = [] ##定义一个用于保存未匹配框的
        not_match_point = []
        # img = cv2.imread('dangxiao_test/3_failedSkeleton.png')
        for j in range(len(seat_result)): #每排4个座位
            min_distance = 500 #定义一个最小距离
            save_flag = False #定义一个用于判断是否保存的标记
            lbp = seat_result[j][0]  #座位左下坐标
            rbp = seat_result[j][1]  #座位右下坐标
            ltp = seat_result[j][3]  #座位左上坐标
            rtp = seat_result[j][2]  #座位右上坐标
            line1 = [lbp, rtp]
            line2 = [rbp, ltp]
            center_piont = self.findIntersection(line1, line2) #find seat center point
            for i in range(skeleton_result.shape[0]):  #（44，25，3）44个人，25个关键点，3维坐标信息x,y,confidence
                x1, y1 = skeleton_result[i][1][:2]  #脖子点
                x6, y6 = skeleton_result[i][6][:2]  #左手肘点
                x8, y8 = skeleton_result[i][8][:2]  #盆骨点 
                if x1==0 or x8==0:  #为0是因为有些人的骨架信息没检测到 
                    continue
                xc, yc = (x1+x8)*0.5, (y1+y8)*0.5
                # cv2.circle(img, tuple([int(xc), int(yc)]), 2, (0, 255, 0), 4)

                if self.isInterArea([xc, yc], seat_result[j]):  #判断点是否在多边形区域内！
                    cv2.circle(img, tuple([int(xc), int(yc)]), 10, (0, 255, 0), 10)
                    save_flag = True
                    vec1 = np.array(center_piont)
                    vec2 = np.array([xc, yc])
                    #计算骨骼点与边框中点欧氏距离
                    distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
                    if distance < min_distance:
                        min_distance = distance #暂存最小距离
                        final_append = i        #暂存最小添加项
                        final = tuple([int(xc), int(yc)])
                    else:
                        not_match_point.append([i, xc, yc])
                    # match.append([i,seat_result[j][4]])  #直接输出第 i 个骨架在第 j 座位
            for k in range(len(failed)):
                if self.isInterArea([failed[k][1],failed[k][2]], seat_result[j]):
                    cv2.circle(img, tuple([int(failed[k][1]), int(failed[k][2])]), 10, (0, 255, 0), 10)
                    save_flag = True
                    vec1 = np.array(center_piont)
                    vec3 = np.array([failed[k][1], failed[k][2]])
                    distance = np.sqrt(np.sum(np.square(vec1 - vec3)))
                    if distance < min_distance:
                        min_distance = distance #暂存最小距离
                        final_append = failed[k][0]  #暂存最小添加项
                        final = tuple([int(failed[k][1]), int(failed[k][2])])
                    else:
                        not_match_point.append(failed[k])
                    # match.append([failed[k][0],seat_result[j][4]]) 
            if save_flag:
                match.append([final_append, seat_result[j][4]]) #将最小项加入座位j
                cv2.circle(img, final, 5, (0, 0, 255), 5)
            else:
                not_match_box.append(seat_result[j])
        return match, not_match_box, not_match_point

    # 找邻近骨骼信息
    def find_neighbor_skeleton(self, skeleton_result):
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
                    neighbor[1] = j
                elif current_distance < submin_distance:
                    submin_distance = current_distance
                    neighbor[2] = j
            result.append(neighbor)
        return result

    # 判断点是否在多边形内
    def isInterArea(self, testPoint, AreaPoint):#testPoint为待测点[x,y]
        LBPoint = AreaPoint[0]#AreaPoint为按顺时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        LTPoint = AreaPoint[1]
        RTPoint = AreaPoint[2]
        RBPoint = AreaPoint[3]
        a = (LTPoint[0]-LBPoint[0])*(testPoint[1]-LBPoint[1])-(LTPoint[1]-LBPoint[1])*(testPoint[0]-LBPoint[0])
        b = (RTPoint[0]-LTPoint[0])*(testPoint[1]-LTPoint[1])-(RTPoint[1]-LTPoint[1])*(testPoint[0]-LTPoint[0])
        c = (RBPoint[0]-RTPoint[0])*(testPoint[1]-RTPoint[1])-(RBPoint[1]-RTPoint[1])*(testPoint[0]-RTPoint[0])
        d = (LBPoint[0]-RBPoint[0])*(testPoint[1]-RBPoint[1])-(LBPoint[1]-RBPoint[1])*(testPoint[0]-RBPoint[0])
        if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0):
            return True
        else:
            return False

    # 此处修改，优化功能求两条线交点，相交返回交点，平行返回[0，0]
    def findIntersection(self, line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]
        try:
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        except ZeroDivisionError:
            point = [0, 0]
        else:
            point = [px, py]
        return point


    # 进行人座匹配，输出 [(骨骼id,列)，行]
    #此处有修改，增加Rematch的功能
    def match(self):
        img = cv2.imread('inputimg/frame_3750.jpg')
        allbox, _ = self.getSeat(img)
        tt = []
        # img = cv2.imread('dangxiao_test/1_paiSeat.png')
        for i in range(len(allbox)):
            _, users = self.getSeat(img)
            sseat = self.aloneSeat(allbox[i])
            for k in range(len(sseat)):
                sseat[k].insert(4, users[i][k])
            tt.append(sseat)
            cv2.line(img, tuple(sseat[0][1]), tuple(sseat[0][2]), (0, 128, 255), 2)
            cv2.line(img, tuple(sseat[1][1]), tuple(sseat[1][2]), (0, 128, 255), 2)
            cv2.line(img, tuple(sseat[2][1]), tuple(sseat[2][2]), (0, 128, 255), 2)
        m = [] #存放每排的 人座匹配（骨骼id，第几个座位）
        not_match_box = []
        not_match_point = []
        for row in range(len(tt)):  # row是第几排=6
            skeleton_result = self.getSkeleton() #由骨架结果计算骨架中心点
            match, ntb, ntp = self.findSite(skeleton_result, tt[row], img) #第几个座位
            m.append(match)
            if len(ntb) != 0:
                for apbox in ntb:
                    not_match_box.append(apbox)
            if len(ntp) != 0:
                for appoint in ntp:
                    not_match_point.append(appoint)
        for idex in range(len(not_match_box)):
            a = not_match_box[idex][0]
            b = not_match_box[idex][1]
            c = not_match_box[idex][2]
            d = not_match_box[idex][3]
            cv2.line(img, tuple(a), tuple(b), (255, 255, 255), 4)
            cv2.line(img, tuple(c), tuple(b), (255, 255, 255), 4)
            cv2.line(img, tuple(c), tuple(d), (255, 255, 255), 4)
            cv2.line(img, tuple(d), tuple(a), (255, 255, 255), 4)
        print(not_match_box)
        print(not_match_point)
        rematch = self.not_match(not_match_box, not_match_point, img)
        print(rematch)
        m.append(rematch)
        cv2.imwrite("result/result-3750.png", img)
        all = []
        all_dict = {}
        for i in range(len(m)):
            for j in range(len(m[i])):
                all.append(m[i][j])
        for k in all:
            all_dict[k[0]] = k[1]
        with open('matchResult2.json', 'w') as f:
            json_str = json.dumps(all_dict, indent=0)
            f.write(json_str)
            f.write('\n')
        
        file = open('matchResult2.json', 'r')
        json_data = json.load(file)
        
        return json_data

    #此处为新增，用于实现rematch
    def not_match(self, not_match_box, not_match_point, img):
        img = img
        skeleton = self.getSkeleton()
        matches = []
        for i in range(len(not_match_box)):
            min_distance = 5000
            save_flag = False
            for j in range(len(not_match_point)):
                x0, y0 = skeleton[not_match_point[j][0]][0][:2]
                xc, yc = not_match_point[j][1:]
                line = [[x0, y0], [xc, yc]]
                flag, center_point = self.line_to_box(not_match_box[i], line, img)
                if flag:
                    save_flag = True
                    vec1 = np.array(center_point)
                    vec2 = np.array([(xc+x0)/2, (yc+y0)/2])
                    distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
                    if distance < min_distance:
                        min_distance = distance  # 暂存最小距离
                        a = not_match_point[j][0]
                        b = not_match_box[i][4]
                        xcc, ycc = not_match_point[j][1:]
            if save_flag:
                matches.append([a, b])
                aa = not_match_box[i][0]
                b = not_match_box[i][1]
                c = not_match_box[i][2]
                d = not_match_box[i][3]
                cv2.line(img, tuple(aa), tuple(b), (255, 255, 0), 4)
                cv2.line(img, tuple(c), tuple(b), (255, 255, 0), 4)
                cv2.line(img, tuple(c), tuple(d), (255, 255, 0), 4)
                cv2.line(img, tuple(d), tuple(aa), (255, 255, 0), 4)
                x00, y00 = skeleton[a][0][:2]
                cv2.line(img, (int(xcc), int(ycc)), (int(x00), int(y00)), (255, 128, 0), 4)
        return matches

    #此处为新增，用于判断线段是否与框相交
    def line_to_box(self, box, line, img):
        img = img
        lbp = box[0]
        rbp = box[1]
        rtp = box[2]
        ltp = box[3]
        lineb = [lbp, rbp]
        liner = [rbp, rtp]
        linet = [ltp, rtp]
        linel = [ltp, lbp]
        line1 = [ltp, rbp]
        line2 = [lbp, rtp]
        center_piont = self.findIntersection(line1, line2)
        b = self.findIntersection(lineb, line)
        r = self.findIntersection(liner, line)
        t = self.findIntersection(linet, line)
        l = self.findIntersection(linel, line)
        if ((int(lbp[0]) < int(b[0]) < int(rbp[0])) and (int(line[0][1]) < int(b[1]) < int(line[1][1]))) \
                or ((int(ltp[0]) < int(t[0]) < int(rtp[0])) and (int(line[0][1]) < int(t[1]) < int(line[1][1]))) \
                or ((int(ltp[1]) < int(l[1]) < int(lbp[1])) and (int(line[0][1]) < int(l[1]) < int(line[1][1]))) \
                or ((int(rtp[1]) < int(r[1]) < int(rbp[1])) and (int(line[0][1]) < int(r[1]) < int(line[1][1]))):
            inters = True
        else:
            inters = False
        return inters, center_piont
'''
原始视频帧：/data/znk/frame_img/cam1/0.jpg
视频帧对应骨架信息：/data/znk/PoseKeypoints/cam1/0.npy
'''
test = ssMatch('javaSeatV2_seat.json', 'skeleton/frame_3750.npy')
json_data = test.match()#运行不变，rematch功能已嵌入match中
print(json_data)
print('save success')

