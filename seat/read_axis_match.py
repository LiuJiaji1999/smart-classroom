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
    print('这条横线是：',least_line) #514，552，1361，736
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

def draw_picture_on_img(img_path, box1, box2, result):
    img = cv2.imread(img_path)
    result.append(box_1)
    result.append(box_2)
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
    cv2.line(img, tuple(box_1[0]), tuple(box_1[1]), color, thickness)
    cv2.line(img, tuple(box_1[1]), tuple(box_1[2]), color, thickness)
    cv2.line(img, tuple(box_1[2]), tuple(box_1[3]), color, thickness)
    cv2.line(img, tuple(box_1[3]), tuple(box_1[0]), color, thickness)
    cv2.line(img, tuple(box_2[0]), tuple(box_2[1]), color, thickness)
    cv2.line(img, tuple(box_2[1]), tuple(box_2[2]), color, thickness)
    cv2.line(img, tuple(box_2[2]), tuple(box_2[3]), color, thickness)
    cv2.line(img, tuple(box_2[3]), tuple(box_2[0]), color, thickness)
    # cv2.line(img, tuple([1237, 271]), tuple([1222, 470]), (225, 225, 0), 3)
    cv2.imwrite("result/temp.jpg", img)

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


def findSite(skeleton_result, seat_result):
    match = []
    for i in range(skeleton_result.shape[0]):  #（44，25，3）44个人，25个关键点，3维坐标信息x,y,confidence
        x1, y1 = skeleton_result[i][1][:2]  #脖子点
        x8, y8 = skeleton_result[i][8][:2]  #盆骨点 
        if x1==0 or x8==0:  #为0是因为有些人的骨架信息没检测到 
            continue
        xc, yc = (x1+x8)*0.5, (y1+y8)*0.5
        # print(xc,yc)
        for j in range(len(seat_result)):
            # print(seat_result[j])
            # xx = site_result[j][:,0]
            # yy = site_result[j][:,1]
            # if xc < min(xx) or xc > max(xx) or yc < min(yy) or yc > max(yy):
            #     continue
            print(j)
            if isInterArea([xc,yc], seat_result[j]):  #判断点是否在多边形区域内！
                match.append((i,j))  #直接输出第 i 个骨架在第 j 排
    return match
           

box_1, box_2 = read_json(".\\bbc08b996f4473319708b127392a0834-asset.json")
seat_result = func3(box_1, box_2, 4) #座位信息
# draw_picture_on_img("1-164317.png", box_1, box_2, seat_result)
skeleton_result = np.load('1-164317.npy') #由骨架结果计算骨架中心点，
# print(len(seat_result))
match = findSite(skeleton_result, seat_result)
print(match)

