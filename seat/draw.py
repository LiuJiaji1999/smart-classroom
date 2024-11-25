from PIL import ImageDraw
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.core.fromnumeric import partition
from scipy import optimize
import json
import cv2



import statistics

list =  [1, 2, 3, 4, 5, 6]
mean = statistics.mean(list)
print(mean)

data = [1, 2, 3, 4, 5, 6]
mean = sum(data)/len(data)
print(mean)

data = [1, 2, 3, 4, 5, 6]
mean = np.mean(data)
print(mean)









# before = {
# "key1": 5,
# "key2": 6,
# "key3": 4,
# "key4": 3,
# }
# # 排序
# after = dict(sorted(before.items(), key=lambda e: e[1]))
# # after = dict(sorted(before.items(), before.keys()))
# print(after)


# path_list = os.listdir('D:/alpha/seat/action/skeleton_16/same')
# for file in path_list:
#     print(file)


# '''
# cv2.polylines(image, [pts], isClosed, color, thickness)>
# 第一个参数image自然就是要标记多边形框的原图，
# pts是多边形框的各个坐标点（这里整个pts参数要注意多加一个中括号，一会儿会结合例子说一下），
# isClosed的参数是是否要将多边形闭合,即最后一个坐标点是否要再连回到第一个坐标点，一会儿会实验给大家看一下差别，
# color是多边形框的颜色，大家选用自己喜欢的颜色就好，

# '''

# # with open('./target/78c373927674b119c255ebf7660cad91-asset.json', 'r') as obj:
# #     dict = json.load(obj)
# # # print(dict['regions'][0]['points'])
# # img = cv2.imread('source/7.jpg')
# # for region in dict['regions']:
# #     # print(region['points'])
# #     points = region['points']
# #     box = []
# #     for i in range(4):
# #         temp = points[i]
# #         point = [int(temp['x']),int(temp['y'])]
# #         box.append(point)
# #     print(box)
# #     pts = np.array(box)
# #     print(pts)
# #     cv2.polylines(img, [pts], True, (0, 0, 255), 2)
# #     cv2.imwrite('result/77.jpg', img)

# '''
# cv2.fillPoly()函数可以用来填充任意形状的图型.可以用来绘制多边形,
# 工作中也经常使用非常多个边来近似的画一条曲线  ---BGR
# .cv2.fillPoly()函数可以一次填充多个图型.

# cv2.fillPoly(img, [pts], color=(0, 0, 255))
   
# cv2.imshow('rect', img) 
# cv2.waitKey(0)
# ''' 


# with open('javaSeatV2_houpai.json', 'r') as obj:
#     dict = json.load(obj)
# img = cv2.imread('houpai.png')
# # print(dict['cameraConfig'])
# # print(len(dict['cameraConfig']))
# areas = []
# for i in range(len(dict['cameraConfig'])):
#     seatConfig = dict['cameraConfig'][i]['seatConfig']
#     # print(seatConfig)
#     for j in range(len(seatConfig)):
#         area = seatConfig[j]['area']
#         areas.append(area)
# # print(areas)
# # # print(len(areas))
# # print(areas[0]) # [{'x': 161, 'y': 719}, {'x': 1148, 'y': 982}, {'x': 1361, 'y': 737}, {'x': 515, 'y': 552}]
# # print(areas[0][0],areas[0][1],areas[0][2],areas[0][3]) #{'x': 161, 'y': 719}
# # print([[areas[0][0]['x'],areas[0][0]['y']],[areas[0][1]['x'],areas[0][1]['y']]]) #[161,719]
# box = []
# for k in range(len(areas)):
#     # print(areas[k])
#     point = [[areas[k][0]['x'],areas[k][0]['y']],[areas[k][1]['x'],areas[k][1]['y']],[areas[k][2]['x'],areas[k][2]['y']],[areas[k][3]['x'],areas[k][3]['y']]]
#     box.append(point)
# # print(len(box))
# # print(box)
# pts = np.array(box)
# print(pts)
# cv2.polylines(img, pts, True, (0, 0, 255), 2)
# cv2.imwrite('77.png', img)



#     # for cameraConfig in dict['cameraConfig']:
#     #     # print(dict['cameraConfig'])
#     #     # print('****************')
#     #     # print(cameraConfig)
#     #     seatConfig = cameraConfig['seatConfig']
#     #     # print(seatConfig)
#     #     for i in range(len(seatConfig)):
#     #         # print(seatConfig[i]['area'])
#     #         area = seatConfig[i]['area']
#     #         areas.append(area)
#     #     # print(areas)
#     #     # print(len(areas))
#     #     # print(len(areas[1]))
#     #     # print(areas[0][0])
#     #     # print(areas[0][1]['x'])
#     #     box = []
#     #     for i in range(len(areas)):  # 2 
#     #         for j in range(len(areas[i])): # 4
#     #             area = [areas[i][j]['x'],area[i][j]['y']]
#     #             box.append(area)
#     #     # print(box)
#     #     pts = np.array(box)
#     #     # print(pts)
#     #     cv2.polylines(img, [pts], True, (0, 0, 255), 2)
#     #     cv2.imwrite('/result/tt.png', img)





# def read_java_json(file_path):
#     with open(file_path) as fp:
#         json_data = json.load(fp)
#     result_1 = json_data["cameraConfig"][0]["seatConfig"][0]["area"]
#     result_2 = json_data["cameraConfig"][0]["seatConfig"][1]["area"]
#     result_3 = json_data["cameraConfig"][1]["seatConfig"][0]["area"]
#     result_4 = json_data["cameraConfig"][1]["seatConfig"][1]["area"]
#     box1 = []
#     box2 = []
#     box3 = []
#     box4 = []
#     for i in range(4):
#         box1.append([result_1[i]['x'],result_1[i]['y']])
#         box2.append([result_2[i]['x'], result_2[i]['y']])
#         box3.append([result_3[i]['x'],result_3[i]['y']])
#         box4.append([result_4[i]['x'], result_4[i]['y']])

#     return box1, box2,box3,box4
# # box1, box2,box3,box4 = read_java_json('javaSeatV2.json')
# # print(box1, box2,box3,box4)




# img = cv2.imread('houpai.png')
# cv2.line(img,(667,469),(1664,517),(0,255,0),5)
# cv2.line(img,(1704,455),(813,409),(0,255,0),5)
# cv2.line(img,(485,539),(1598,601),(0,0,255),5)
# cv2.line(img,(1664,522),(660,471),(0,0,255),5)
# cv2.imwrite('tt.png',img)



# img = np.ones((512,512,3)) #白色背景
# color=(0,255,0)  #绿色
# # pts = np.array([[10,50],[100,50],[100,100],[10,100]])
# pts2 = np.array([[10,50],[10,100],[100,50],[100,100]])
# # pts = pts.reshape((-1,1,2))
# pts2 = pts2.reshape((-1,1,2))
# # cv2.polylines(img,[pts],True,color,3)
# cv2.polylines(img,[pts2],True,color,3)
# cv2.imshow('juzicode.com',img)
# cv2.waitKey()


# a = [2, 3, 4, 5]
# b = [2, 3, 4]

# # tmp = [val for val in a if val in b]
# # print(tmp)
# # [2, 5]

# print (list(set(a).intersection(set(b)))) #交集
# print (list(set(a).union(set(b)))) #并集
# print (list(set(a).difference(set(b)))) # a中有而b中没有的  差集

# tmp = [val for val in a if val not in b]
# print(tmp)


# import json


# def writeDict(data):
#     with open("data.txt", "w") as f:
#     	f.write(json.dumps(data, ensure_ascii=False))

# if __name__ == '__main__':

#     dict_1 = {"北京": "BJP", "北京北": "VAP", "北京南": "VNP", "北京东": "BOP", "北京西": "BXP"}
#     print(dict_1)
#     writeDict(dict_1)


# max_d = {"scholl":'123'}
# print(*max_d)
# print(**max_d)

'''
列表前面加星号作用是将列表解开成两个独立的参数，传入函数，

字典前面加两个星号，是将字典解开成独立的元素作为形参。
'''
# def add(a, b):
#     return a+b
 
# data = [4,3]
# print(*data)
# print(add(*data))
# #equals to print add(4, 3)
# data = {'a' : 4, 'b' : 3}
# print(**data)
# print(add(**data))
# #equals to print add(4, 3)

# a=[1,2,3,4,5]

# print(a)                    #打印整个数组----->[1, 2, 3, 4, 5]

# #一个参数
# print(a[-1])                #取最后一个元素----->5

# #二个参数
# print(a[:-1])               #除了最后一个取全部----->[1, 2, 3, 4]
# print(a[1:])                #取第二个到最后一个元素----->[2, 3, 4, 5]

# #三个参数
# print(a[::-1])              #取从后向前的元素----->[5, 4, 3, 2, 1]
# print(a[2::-1])             #取从下标为2的元素,从后向前----->[3, 2, 1]
# print(a[:2])                #取从前往后的前两个数组------>[1, 2]
# print(a[:3]) 


# import pickle
# # path = '/data/liujiaji/action/c1_back_dispersed.pkl'   #path='/root/……/aus_openface.pkl'   pkl文件所在路径
# path = '/data/workspace/yaojiapeng/dangxiao/smartcam/weights/pose_svm_dangxiao_v1.pkl'   
# f=open(path,'rb')
# data=pickle.load(f)
 
# print(data)