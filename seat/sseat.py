# 1. 每4个点拟合成一条线， 
# 2. 每条线自己做分割
# 每个点的四个坐标按照从左下开始逆时针记录
import re
import numpy as np
from scipy import optimize
import json
import cv2
def optimize_line(x, y):
    def residuals(p):
        k, b = p
        return y - (k*x + b) 
    r = optimize.leastsq(residuals, [1, 0])
    k, b = r[0]
    return k, b

def read_json(file_path):
    with open(file_path) as fp:
        json_data = json.load(fp)
    result_1 = json_data["regions"][0]["points"]
    result_2 = json_data["regions"][1]["points"]
    box1 = []
    box2 = []
    for i in range(4):
        temp = result_1[i]
        point = [temp['x'], temp['y']]
        box1.append(point)
        box2.append([result_2[i]['x'], result_2[i]['y']])
    return box1, box2

def read_java_json(file_path):
    with open(file_path) as fp:
        json_data = json.load(fp)
    result_1 = json_data["cameraConfig"][0]["seatConfig"][0]["area"]
    result_2 = json_data["cameraConfig"][0]["seatConfig"][1]["area"]
    box1 = []
    box2 = []
    for i in range(4):
        temp = result_1[i]
        point = [temp['x'], temp['y']]
        box1.append(point)
        box2.append([result_2[i]['x'], result_2[i]['y']])
    return box1, box2

def func2(box1, box2, n, n1=None):
    # x,y 分别是每个点的x，y坐标，顺序为[左下，右下，右上，左上]
    # n为两排之间存在多少排座位
    # 输出中间n排座位的四个点坐标
    line1_x = np.array([box1[0][0], box1[3][0], box2[0][0], box2[3][0]])
    line1_y = np.array([box1[0][1], box1[3][1], box2[0][1], box2[3][1]])
    line2_x = np.array([box1[1][0], box1[2][0], box2[1][0], box2[2][0]])
    line2_y = np.array([box1[1][1], box1[2][1], box2[1][1], box2[2][1]])
    k1, b1  = optimize_line(line1_x, line1_y)
    k2, b2  = optimize_line(line2_x, line2_y)

    #计算变化比例
    rate = pow((line1_x[3] - line1_x[2])/(line1_x[1] - line1_x[0]), 1/(n+1))
    rate2 = pow((line2_x[3] - line2_x[2])/(line2_x[1] - line2_x[0]), 1/(n+1))

    least_line = [box1[3], box1[2]]
    result = []
    for i in range(n-1):
        x1 = (rate**(i+1))*(box1[3][0] - box1[0][0]) + least_line[0][0]
        y1 = k1*x1 + b1
        x2 = (rate2**(i+1))*(box1[2][0] - box1[1][0]) + least_line[1][0]
        y2 = k2*x2 + b2
        new_box = [least_line[0], least_line[1], [x2, y2], [x1, y1]]
        least_line = [[x1, y1], [x2, y2]]
        result.append(new_box)
    result.append([least_line[0], least_line[1], box2[1], box2[0]])
    if n1:
        least_line = [box2[3], box2[2]]
        for i in range(n1):
            x1 = (rate**(i+1+n))*(box1[3][0] - box1[0][0]) + least_line[0][0]
            y1 = k1*x1 + b1
            x2 = (rate2**(i+1+n))*(box1[2][0] - box1[1][0]) + least_line[1][0]
            y2 = k2*x2 + b2
            new_box = [least_line[0], least_line[1], [x2, y2], [x1, y1]]
            least_line = [[x1, y1], [x2, y2]]
            result.append(new_box)
    return result

def func3(box1, box2, n, n1=None):
 
    # x,y 分别是每个点的x，y坐标，顺序为[左下，右下，右上，左上]
    # n为两排之间存在多少排座位   n1表示 座位2后还有几排？？
    # 输出中间n排座位的四个点坐标
    # 最后返回 所有座位（包含标注好的、比例计算生成的）
    line1_x = np.array([box1[0][0], box1[3][0], box2[0][0], box2[3][0]])
    line1_y = np.array([box1[0][1], box1[3][1], box2[0][1], box2[3][1]])
    line2_x = np.array([box1[1][0], box1[2][0], box2[1][0], box2[2][0]])
    line2_y = np.array([box1[1][1], box1[2][1], box2[1][1], box2[2][1]])
    # print('这条线1的x坐标:',line1_x)
    # print('这条线1的y坐标:',line1_y)
    # print('这条线2的x坐标:',line2_x)
    # print('这条线2的y坐标:',line2_y)

    k1, b1  = optimize_line(line1_x, line1_y)
    k2, b2  = optimize_line(line2_x, line2_y)

    #计算变化比例
    #3           1218，202 ---------------1819，248
    #2         1136，239 ---------------1765，302
    #                  ---------------
    #                ---------------
    #              ---------------
    #1  514，552 ---------------1361，736
    #0 160，719---------------1148，981
    rate = pow((line1_x[3] - line1_x[2])/(line1_x[1] - line1_x[0]), 1/(n+1)) #为啥进行求幂运算？？
    rate2 = pow((line2_x[3] - line2_x[2])/(line2_x[1] - line2_x[0]), 1/(n+1))
    distance1 = line1_x[2] - line1_x[1] # 中间间排座位的竖向距离 
    distance2 = line2_x[2] - line2_x[1]
    denom1, denom2 = 0, 0
    for i in range(n):  # 间隔 n 排
        denom1 += rate**(i+1) 
        denom2 += rate**(i+1)
    least_line = [box1[3], box1[2]]
    # print('这条横线是：',least_line) #514，552，1361，736
    seat = [box1] #先把标注好的座位1 追加进去，然后后续循环骨架点，找位置
    for i in range(n-1):
        x1 = distance1*(rate**(i+1))/denom1
        x1 += least_line[0][0]
        y1 = k1*x1 + b1

        x2 = distance2*(rate2**(i+1))/denom2
        x2 += least_line[1][0]
        y2 = k2*x2 + b2

        new_box = [least_line[0], least_line[1], [x2, y2], [x1, y1]]
        least_line = [[x1, y1], [x2, y2]]
        seat.append(new_box) #循环 每次追加新的一排 也就是new_box
    seat.append([least_line[0], least_line[1], box2[1], box2[0]]) #最后追加一个大盒子，
    seat.append(box2) #最后把标注好的最后排座位 追加，为了后续的骨架遍历
    if n1:   # 表示n1非空时！
        least_line = [box2[3], box2[2]] #标注好的座位2的最后一条线
        for i in range(n1):
            x1 = (rate**(i+1+n))*(box1[3][0] - box1[0][0]) + least_line[0][0]
            y1 = k1*x1 + b1

            x2 = (rate2**(i+1+n))*(box1[2][0] - box1[1][0]) + least_line[1][0]
            y2 = k2*x2 + b2

            new_box = [least_line[0], least_line[1], [x2, y2], [x1, y1]]
            least_line = [[x1, y1], [x2, y2]]
            seat.append(new_box)

    # seat = np.array(seat)
    return seat

'''
with open('javaSeatV2.json', 'r') as obj:
    dict = json.load(obj)
# print(dict['regions'][0]['points'])
img = cv2.imread('1-164317.png')
# print(dict['cameraConfig'])
areas = []
for cameraConfig in dict['cameraConfig']:
    seatConfig = cameraConfig['seatConfig']
    for i in range(len(seatConfig)):
        # print(seatConfig[i]['area'])
        area = seatConfig[i]['area']
        areas.append(area)
    # print(areas)
    # print(len(areas))
    # print(len(areas[1]))
    # print(areas[0][0])
    # print(areas[0][1]['x'])
    box = []
    for i in range(len(areas)):  # 2 
        for j in range(len(areas[i])): # 4
            area = [areas[i][j]['x'],area[i][j]['y']]
            box.append(area)
    # print(box)
    pts = np.array(box)
    # print(pts)
    cv2.polylines(img, [pts], True, (0, 0, 255), 2)
    cv2.imwrite('/result/tt.png', img)
'''

# 4 在排排座位上，切分各个座位
def aloneSeat1(box1): 
    for i in range(4):
        for j in range(2):
            box1[i][j] = int(box1[i][j]) 
    
    line1 = [box1[0], box1[1]]
    dis1_x = line1[1][0] - line1[0][0]
    dis1_y = line1[1][1] - line1[0][1]
    seat_dis1_x = dis1_x/4
    seat_dis1_y = dis1_y/4
    # print(line1)
    # print('这条线1共多长：',dis1_x)
    # print('1p每个座位平均x间隔：',seat_dis1_x)
    # print(box1[0],box1[1],box1[2],box1[3]) #[160, 719] [1148, 981] [1361, 736] [514, 552]
    relt1 = [box1[0]]  # [[160, 719]]
    x,y = line1[0][0],line1[0][1]
    # print(relt1)
    for i in range(4):
        x += seat_dis1_x
        y += seat_dis1_y
        x = int(x)
        y = int(y)
        new_seat = [x,y]
        relt1.append(new_seat)
        # relt1.insert()
    # print('1p每个座位坐标：',relt1)
    # print('1p1g座位坐标：',relt1[0])
    # print('1p2g座位坐标：',relt1[1])
    # print('1p3g座位坐标：',relt1[2])

    # print(box1[0],box1[1],box1[2],box1[3])
    line2 = [box1[3], box1[2]]
    dis2_x = line2[1][0] - line2[0][0]
    dis2_y = line2[1][1] - line2[0][1]
    seat_dis2_x = dis2_x/4
    seat_dis2_y = dis2_y/4
    # print('这条线2共多长：',dis2_x)
    # print('2p每个座位平均x间隔：',seat_dis2_x)
    relt2 = [box1[3]]
    x,y = line2[0][0],line2[0][1]
    for i in range(4):
        x += seat_dis2_x
        y += seat_dis2_y
        x = int(x)
        y = int(y)
        line = [x,y]
        relt2.append(line)
    # print('2p每个座位坐标：',relt2)
    # print('2p1g座位坐标：',relt2[0])
    # print('2p2g座位坐标：',relt2[1])
    # print('2p3g座位坐标：',relt2[2])

    img = cv2.imread('school/paipai.png')
    cv2.line(img, tuple(relt1[1]), tuple(relt2[1]),(255,0,0),3)
    cv2.line(img, tuple(relt1[2]), tuple(relt2[2]),(255,0,0),3)
    cv2.line(img, tuple(relt1[3]), tuple(relt2[3]),(255,0,0),3)
    cv2.imwrite("school/diyipai.png", img)

    # print(box1[0],box1[1],box1[2],box1[3])
    ss = [] #第一排所有座位
    ss1 = [relt1[0],relt1[1],relt2[1],relt2[0]]
    ss2 = [relt1[1],relt1[2],relt2[2],relt2[1]]
    ss3 = [relt1[2],relt1[3],relt2[3],relt2[2]]
    ss4 = [relt1[3],relt1[4],relt2[4],relt2[3]]
    ss.append(ss1)
    ss.append(ss2)
    ss.append(ss3)
    ss.append(ss4)

    #划分座位的同时，获取人员id
    with open('javaSeatV2.json', 'r') as obj:
        dict = json.load(obj)
    stu_1 =  dict["cameraConfig"][0]["seatConfig"][0]["userIdList"]
    stu_2 =  dict["cameraConfig"][0]["seatConfig"][1]["userIdList"]
    ps = list(zip(stu_1,ss))

    pss = []
    for i in range(len(ss)):
        # print(stu_1[i],ss[i])
        pss.append((stu_1[i],i))
    # print(pss)
    
    return ss

'''
测试新数据的座位划分情况
'''
# bx1 = read_json("target/78c373927674b119c255ebf7660cad91-asset.json")
# s = aloneSeat1(bx1)


'''
排排座位的坐标信息
[1218  202] [1819  248] 53 52
[1136  239] [1765  302] 50 51
[1041  288] [1716  357] 40 41
[ 914  351] [1628  453]  30 31
[ 743  435] [1513  578]  20 21
[ 514  552] [1361  736] 10 11
[ 160  719] [1148  981] 00 01
'''
def aloneSeat2(seat): 
    #先得到每排座位的线
    lineR = []
    for i in range(len(seat)):
        line = [seat[i][0],seat[i][1]]
        lineR.append(line)
    lineR.append([seat[len(seat)-1][3],seat[len(seat)-1][2]])
    # print(len(lineR))
    # print(lineR)
    # print(np.array(lineR))
    # print(lineR[0])  # 第一条线 [[160, 719], [1148, 981]]
    # print(lineR[0][0]) # 第一条线的 xy坐标 [160, 719]
    # print(lineR[0][0][0]) # 第一条线的 x坐标 160
    # 计算每条线的 x y差值并 等分4
    ll = [] # 存放新生成的坐标点
    lineAll = []
    for i in range(len(lineR)): #0-6 共7条线
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
    print(ase.shape) 

    img = cv2.imread('school/temp.png')
    for i in range(len(ll)):
        cv2.line(img, tuple(ll[i]), tuple(ll[i+4]),(255,0,0),3)
        i += 1
        if i == len(ll)-4:
            break  
    cv2.imwrite("school/seats.png", img)

# 3 在骨骼点图上，补全排排的座位信息
def draw_picture_on_img(img_path, box1, box2, result):
    img = cv2.imread(img_path)
    # result.append(box_1)
    # result.append(box_2)
    for box in result:
        a = list(np.random.choice(range(256), size=3))
        color = (0, 0, 225)
        # print(color)
        thickness = 4
        for x in box:
            x[0] = int(x[0])
            x[1] = int(x[1])
        cv2.line(img, tuple(box[0]), tuple(box[1]), color, thickness)
        cv2.line(img, tuple(box[1]), tuple(box[2]), color, thickness)
        cv2.line(img, tuple(box[2]), tuple(box[3]), color, thickness)
        cv2.line(img, tuple(box[3]), tuple(box[0]), color, thickness)
    color = (0, 225, 0)
    # print(tuple(box_1[0]))
    cv2.line(img, tuple(box_1[0]), tuple(box_1[1]), color, thickness)
    cv2.line(img, tuple(box_1[1]), tuple(box_1[2]), color, thickness)
    cv2.line(img, tuple(box_1[2]), tuple(box_1[3]), color, thickness)
    cv2.line(img, tuple(box_1[3]), tuple(box_1[0]), color, thickness)
    cv2.line(img, tuple(box_2[0]), tuple(box_2[1]), color, thickness)
    cv2.line(img, tuple(box_2[1]), tuple(box_2[2]), color, thickness)
    cv2.line(img, tuple(box_2[2]), tuple(box_2[3]), color, thickness)
    cv2.line(img, tuple(box_2[3]), tuple(box_2[0]), color, thickness)
    # cv2.line(img, tuple([1237, 271]), tuple([1222, 470]), (225, 225, 0), 3)

    cv2.imwrite("school/paipai.png", img)


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
# 1、 先画映射的骨骼点
def yinshe_skeleton(skeleton_result):
    relation = find_neighbor_skeleton(skeleton_result)
    # print(relation)
    # print(len(relation))
    img = cv2.imread('1-164317.png')
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
    cv2.imwrite('school/show_failed_result.png', img)
    return failed


def isInterArea(testPoint,AreaPoint):#testPoint为待测点[x,y]
    LBPoint = AreaPoint[0]#AreaPoint为按顺时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    LTPoint = AreaPoint[1]
    RTPoint = AreaPoint[2]
    RBPoint = AreaPoint[3]
    a = (LTPoint[0]-LBPoint[0])*(testPoint[1]-LBPoint[1])-(LTPoint[1]-LBPoint[1])*(testPoint[0]-LBPoint[0])
    b = (RTPoint[0]-LTPoint[0])*(testPoint[1]-LTPoint[1])-(RTPoint[1]-LTPoint[1])*(testPoint[0]-LTPoint[0])
    c = (RBPoint[0]-RTPoint[0])*(testPoint[1]-RTPoint[1])-(RBPoint[1]-RTPoint[1])*(testPoint[0]-RTPoint[0])
    d = (LBPoint[0]-RBPoint[0])*(testPoint[1]-RBPoint[1])-(LBPoint[1]-RBPoint[1])*(testPoint[0]-RBPoint[0])
    #print(a,b,c,d)
    if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
        return True
    else:
        return False

# 2 、在映射好的骨骼点上，补全其他本身找到的点，进而得到所有的骨骼点图
def findSite(skeleton_result, seat_result):
    failed = yinshe_skeleton(skeleton_result) #得到映射后的xc.yc坐标，进而补充所有的xc,yc，进行人座匹配
    match = []
    img = cv2.imread('school/show_failed_result.png')
  
    for j in range(len(seat_result)):
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
            if isInterArea([xc,yc], seat_result[j]):  #判断点是否在多边形区域内！
                match.append((i,j))  #直接输出第 i 个骨架在第 j 排
        for k in range(len(failed)):
            if isInterArea([failed[k][1],failed[k][2]], seat_result[j]):  #判断点是否在多边形区域内！
                match.append((failed[k][0],j))  #直接输出第 i 个骨架在第 j 排   
    cv2.imwrite('school/show_all_skeleton.png',img)
    return match
           


# box_1,box_2 = read_json("bbc08b996f4473319708b127392a0834-asset.json")
box_1,box_2 = read_java_json('javaSeatV2.json')


seat_result = func3(box_1, box_2, 4) #座位信息
# for i in range(len(seat_result)):
#     for j in range(4):
#         for k in range(2):
            # seat_result[i][j][k] = int(seat_result[i][j][k])
# print(np.array(seat_result))
# print(seat_result[0])   # 第一排座位
# print(seat_result[0][0])      # 第一排座位的左下坐标   
# print(seat_result[0][1]) 
# print(seat_result[0][0][0])    # 第一排座位的左下坐标 的x值

seats = aloneSeat1(box_1)
# sseat = aloneSeat2(seat_result)

# draw_picture_on_img("school/show_all_skeleton.png", box_1, box_2, seat_result)

skeleton_result = np.load('1-164317.npy') #由骨架结果计算骨架中心点，
# match = findSite(skeleton_result, seat_result) #第几排
match = findSite(skeleton_result, seats) #第几个座位
print(match)
