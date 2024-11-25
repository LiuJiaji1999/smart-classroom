import os
import numpy as np
import json
import cv2
from datetime import datetime

class ssMatch(object):
    def __init__(self, img_path,json_path,skeleton_path,save_img): 
         self.json_path = json_path
         self.skeleton_path = skeleton_path
         self.img_path = img_path
         self.save_img = save_img

    # 1 读取座位标注，划分排排的座位信息          
    def getSeat(self):
        img = cv2.imread(self.img_path)
        with open(self.json_path, 'r') as obj:
            dict = json.load(obj)
        try:
            seatsort = dict['seatsort']
        except:
            seatsort = 'rtol'
        users = []
        areas = []
        seatnums = []
        for i in range(len(dict['cameraConfig'])):
            seatConfig = dict['cameraConfig'][i]['seatConfig']
            for j in range(len(seatConfig)):
                area = seatConfig[j]['seatListArea']
                areas.append(area)
                userId = seatConfig[j]['userIdList']
                users.append(userId)
                seatnum = seatConfig[j]['seatNum']
                seatnums.append(seatnum)
        seat = [] #排排座位
        for k in range(len(areas)):
            point = [[areas[k][0]['x'],areas[k][0]['y']],[areas[k][1]['x'],areas[k][1]['y']],[areas[k][2]['x'],areas[k][2]['y']],[areas[k][3]['x'],areas[k][3]['y']]]
            point.sort(key=lambda x: x[0], reverse=False)
            point = [point[0],point[2],point[3],point[1]] #右上视角：逆时针、左上视角：顺时针s
            seat.append(point)
        # print(seat)
        pts = np.array(seat) 
        # print(pts)
        cv2.polylines(img, pts, True, (0, 0, 255), 2)
        # cv2.imwrite("dangxiao_test/1_paiSeat.png", img)
        cv2.imwrite(self.save_img, img)

        return seat,users,seatnums,seatsort
     
    # 2 读取检测到的骨架信息
    def getSkeleton(self):
        skeleton_result = np.load(self.skeleton_path)  #由骨架结果计算骨架中心点
        return skeleton_result


    
    # 3 在排排座位上，切分各个座位
    def aloneSeat(self,seat,seatnum,seatsort): 
        ss = [] #存放每排座位
        line1 = [seat[0], seat[1]]
        # print(line1)
        dis1_x = line1[1][0] - line1[0][0]
        dis1_y = line1[1][1] - line1[0][1]
        seat_dis1_x = dis1_x/4
        seat_dis1_y = dis1_y/4
        relt1 = [seat[0]]  # [[160, 719]]
        x,y = line1[0][0],line1[0][1]
        for i in range(seatnum):
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
        for i in range(seatnum):
            x += seat_dis2_x
            y += seat_dis2_y
            x = int(x)
            y = int(y)
            line = [x,y]
            relt2.append(line)
        

        # ss0 = [relt1[0],relt1[1],relt2[1],relt2[0]]
        # ss1 = [relt1[1],relt1[2],relt2[2],relt2[1]]
        # ss2 = [relt1[2],relt1[3],relt2[3],relt2[2]]
        # ss3 = [relt1[3],relt1[4],relt2[4],relt2[3]]
        
        # if seatsort == 'rtol':
        #     ss = [ss3,ss2,ss1,ss0]  #从右到左 每一排所有座位          
        # else:
        #     ss = [ss0,ss1,ss2,ss3]  #从左到右   每一排所有座位
        for i in range(seatnum):
            s = [relt1[i],relt1[i+1],relt2[i+1],relt2[i]]
            ss.append(s)  #从左到右   每一排所有座位
        if seatsort == 'rtol':  #从右到左 每一排所有座位   
            ss.reverse() 
        return ss

    #  4 映射未检测到的骨骼信息
    def yinshe_skeleton(self,skeleton_result):
        # img = cv2.imread('dangxiao_test/2_aloneSeat.png')
        img = cv2.imread(self.save_img)
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
            
            cv2.circle(img, tuple([int(xc),int(yc)]),2,(0,255,0), 4)
        # cv2.imwrite('dangxiao_test/3_failedSkeleton.png', img)
        cv2.imwrite(self.save_img, img)
        return failed

    

    # 5 在映射好的骨骼点上，补全其他本身找到的点，进而得到所有的骨骼点图
    def findSite(self,skeleton_result,seat_result):
        failed = self.yinshe_skeleton(skeleton_result) #得到映射后的xc.yc坐标，进而补充所有的xc,yc，进行人座匹配
        # all = [] 
        match = []
        not_match_box = [] ##定义一个用于保存未匹配座位框的
        not_match_point = [] ##保存未匹配的骨骼点 [id,xc,yc]
        # img = cv2.imread('dangxiao_test/3_failedSkeleton.png')
        img = cv2.imread(self.save_img)
        for j in range(len(seat_result)): #每排4个座位 j是第几个座位
            # alls = [j,seat_result[j]] #每排座位的信息 j排 四个点坐标，用户id
            # all.append(alls)

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
                x8, y8 = skeleton_result[i][8][:2]  #盆骨点 
                if x1==0 or x8==0:  #为0是因为有些人的骨架信息没检测到 
                    continue
                xc, yc = (x1+x8)*0.5, (y1+y8)*0.5
                cv2.circle(img, tuple([int(xc),int(yc)]),2,(0,255,0), 4) 

                if self.isInterArea([xc,yc], seat_result[j]):  #判断点是否在多边形区域内！
                    save_flag = True
                    vec1 = np.array(center_piont) #座位的中心点坐标 
                    vec2 = np.array([xc, yc]) #骨骼点
                    #计算骨骼点与边框中点欧氏距离
                    distance = np.sqrt(np.sum(np.square(vec1 - vec2)))  #二者的欧氏距离
                    if distance < min_distance:
                        min_distance = distance #暂存最小距离
                        final_append = i        #暂存最小添加项
                        # final_append = [j, i]        #暂存最小添加项
                    else:
                        not_match_point.append([i, xc, yc])  
                    # match.append((j,i,seat_result[j][4]))  #直接输出第 i 个骨架在第 j 座位
                    # match.append([i,seat_result[j][4]])  #直接输出第 i 个骨架在第 j 座位
            for k in range(len(failed)):
                if self.isInterArea([failed[k][1], failed[k][2]], seat_result[j]):
                    save_flag = True
                    vec1 = np.array(center_piont)
                    vec3 = np.array([failed[k][1], failed[k][2]])
                    distance = np.sqrt(np.sum(np.square(vec1 - vec3)))
                    if distance < min_distance:
                        min_distance = distance #暂存最小距离
                        final_append = failed[k][0]  #暂存最小添加项
                    else:
                        not_match_point.append(failed[k])
                        # final_append = [j,failed[k][0]]  #暂存最小添加项
                    # match.append([failed[k][0], seat_result[j][4]])
            if save_flag:
                match.append([final_append, seat_result[j][4]]) #将最小项加入座位j
                # final_append.insert(2,seat_result[j][4])
                # match.append(final_append)
            else:
                not_match_box.append(seat_result[j])
        # cv2.imwrite('dangxiao_test/4_allMatch.png',img) 
        cv2.imwrite(self.save_img, img)

        return match,not_match_box, not_match_point

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
                if current_distance < min_distance:
                    min_distance = current_distance
                    neighbor[1] = j #
                elif current_distance < submin_distance:
                    submin_distance = current_distance
                    neighbor[2] = j
            result.append(neighbor)
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
    
    # 优化功能 求两条线交点，相交返回交点，平行返回[0，0]
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

   # 实现rematch
    def not_match(self, not_match_box, not_match_point):
        # img = cv2.imread('psmatch/4_c4bf_allMatch.png')
        skeleton = self.getSkeleton()
        matches = []
        for i in range(len(not_match_box)):
            min_distance = 5000
            save_flag = False
            for j in range(len(not_match_point)):
                x0, y0 = skeleton[not_match_point[j][0]][0][:2]
                xc, yc = not_match_point[j][1:]
                line = [[x0, y0], [xc, yc]]
                flag, center_point = self.line_to_box(not_match_box[i], line)
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
                # aa = not_match_box[i][0]
                # b = not_match_box[i][1]
                # c = not_match_box[i][2]
                # d = not_match_box[i][3]
                # cv2.line(img, tuple(aa), tuple(b), (255, 255, 0), 4)
                # cv2.line(img, tuple(c), tuple(b), (255, 255, 0), 4)
                # cv2.line(img, tuple(c), tuple(d), (255, 255, 0), 4)
                # cv2.line(img, tuple(d), tuple(aa), (255, 255, 0), 4)
                # x00, y00 = skeleton[a][0][:2]
                # cv2.line(img, (int(xcc), int(ycc)), (int(x00), int(y00)), (255, 128, 0), 4)
        return matches

     # 判断线段是否与框相交
    def line_to_box(self, box, line):
        # img = img
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

    # 进行人座匹配，输出 [(骨骼id,列)，行] 
    def match(self,json_file):
        allbox,users,seatnums,seatsort = self.getSeat()
        tt = []
        # img = cv2.imread('dangxiao_test/1_paiSeat.png')
        img = cv2.imread(self.save_img)
        for i in range(len(allbox)): #几排
            sseat = self.aloneSeat(allbox[i],seatnums[i],seatsort) # 按每排 seatnums[i] 个划分 allbox[i] 排
            for k in range(len(sseat)): #一排的座位
                sseat[k].insert(4,users[i][k])
            tt.append(sseat)
            # cv2.line(img, tuple(sseat[0][1]), tuple(sseat[0][2]),(255,0,0),3)
            # cv2.line(img, tuple(sseat[1][1]), tuple(sseat[1][2]),(255,0,0),3)
            # cv2.line(img, tuple(sseat[2][1]), tuple(sseat[2][2]),(255,0,0),3)

            cv2.line(img, tuple(sseat[0][0]), tuple(sseat[0][3]),(255,0,0),3)
            cv2.line(img, tuple(sseat[1][0]), tuple(sseat[1][3]),(255,0,0),3)
            cv2.line(img, tuple(sseat[2][0]), tuple(sseat[2][3]),(255,0,0),3)

            # cv2.imwrite("dangxiao_test/2_aloneSeat.png", img)
            cv2.imwrite(self.save_img, img)
        print(sseat)
        # return tt
        # p = [] #存放第几排
        m = [] #存放每排的 人座匹配（骨骼id，第几个座位）
        # nm = [] #存放未匹配到人的座位信息
        # a = [] #存放 追加所有座位的信息
        not_match_box = []
        not_match_point = []
        for l in range(len(tt)): # l是第几排
            skeleton_result = self.getSkeleton() #由骨架结果计算骨架中心点
            match,ntb,ntp = self.findSite(skeleton_result,tt[l]) #第几个座位
            # p.append(l)
            m.append(match) # 最终结果
            # a.append(all)
            if len(ntb) != 0:
                for apbox in ntb:
                    not_match_box.append(apbox)
            if len(ntp) != 0:
                for appoint in ntp:
                    not_match_point.append(appoint)
        rematch = self.not_match(not_match_box, not_match_point)
        # result = dict(zip(p,m))
        # print('未匹配人与框',rematch)
        m.append(rematch)
        allm = []    
        dict = {}
        for i in range(len(m)):
            for j in range(len(m[i])):
                allm.append(m[i][j])    
        for k in allm:
            dict[k[0]] = k[1]
        # with open('action/match/c1_front_1710.json', 'w') as f:
        with open(json_file, 'w') as f:
            json_str = json.dumps(dict,indent=0)
            f.write(json_str)
            f.write('\n')   
        file = open(json_file,'r')
        match_json_data = json.load(file)

        return match_json_data
        # return match_json_data,allm

    def match_sekeleton_behavior(self,json_file):
        _,allm = self.match(json_file)
        bs=[['a','举手'],['b',''],['c',''],['d','举手'],
['e','玩手机'],['f',''],['g',''],['h','举手'],
['i',''],['j',''],['k','其他'],['l',''],
['m','举手'],['n',''],['o','举手'],['p',''],
['q',''],['r',''],['s',''],['t','举手'],
['u','举手'],['v',''],['w',''],['x','举手']]
        dict2 = {}
        for k in bs:
            dict2[k[0]] = k[1]

        sebe = []
        for key in dict2:
            for i in range(len(allm)):
                if allm[i][1] == key:
                    # print (key, dict2[key])
                    sebe.append([allm[i][0], dict2[key]]) #骨骼id-行为
        
        #存放字典
        dict = {}
        for k in sebe:
            dict[k[0]] = k[1]
        with open(json_file, 'w') as f:
            json_str = json.dumps(dict,indent=0)
            f.write(json_str)
            f.write('\n')   
        # file = open('action/sebe/c1_front_1710.json','r',encoding='utf-8')
        file = open(json_file,'r',encoding='utf-8')
        sebe_json_data = json.load(file)
        return sebe_json_data


      
'''
原始视频帧：/data/znk/frame_img/cam1/0.jpg
视频帧对应骨架信息：/data/znk/PoseKeypoints/cam1/0.npy
'''
# path_list = os.listdir('D:/alpha/seat/action/dxtest_11/temp/')
# for file in path_list:
#     test = ssMatch('action/dxtest_11/temp/%s.jpg' % (os.path.splitext(file)[0]) ,'action/dxtest_10shangwu/c4_back_action_houduan.json','action/dxtest_11/npy/%s.npy' % (os.path.splitext(file)[0]),'action/dxtest_11/final_img/%s.jpg' % (os.path.splitext(file)[0]))
#     match_json_data = test.match('action/dxtest_11/match/%s.json' % (os.path.splitext(file)[0]))
#     # sebe_json_data = test.match_sekeleton_behavior('action/tran/tran_sebe/%s.json' % (os.path.splitext(file)[0]))
# print('人座匹配结果：',match_json_data)
# # print('未匹配结果：',nm)
# # print('骨骼-行为：',sebe_json_data)
# print('success')

# #单张图片   20220216095200_0b1de073dcf17a7423b6dbb7dd836215  c3_back_action_houduan
test = ssMatch('action/dxtest_11/temp/20220216095200_b6deed9e6041a5d16a24ea99065e6cea.jpg' ,'action/dxtest_11/c4_back_action.json','action/dxtest_11/npy/20220216095200_b6deed9e6041a5d16a24ea99065e6cea.npy','action/dxtest_11/final_img/20220216095200_b6deed9e6041a5d16a24ea99065e6cea.jpg')
# test = ssMatch('action/dxtest_11/img/20220216095200_0b1de073dcf17a7423b6dbb7dd836215.jpg' ,'action/dxtest_11/c3_back_action.json','action/dxtest_11/npy/20220216095200_0b1de073dcf17a7423b6dbb7dd836215.npy','action/dxtest_11/final_img/20220216095200_0b1de073dcf17a7423b6dbb7dd836215.jpg')

match_json_data = test.match('action/dxtest_11/match/change1.json')
print('人座匹配结果：',match_json_data)
# print('未匹配结果：',nm)
# print('骨骼-行为：',sebe_json_data)
print('success')


# bs=[[['a','其他'],['b','玩手机'],['c',''],['d','玩手机']],
#     [['e','玩手机'],['f','其他'],['g','玩手机'],['h','记笔记']],
#     [['i',''],['j','睡觉'],['k','玩手机'],['l','玩手机']],
#     [['m','玩手机'],['n','玩手机'],['o','玩手机'],['p','其他']],
#     [['q','其他'],['r','其他'],['s',''],['t','其他']]]

