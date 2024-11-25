import json
from PIL import Image
import re
import numpy as np
import sklearn
import math
import sys
sys.path.append('/data/liujiaji/action/smartcam')
from linkeaction import SinglePersonSVM
from sklearn.metrics import classification_report,confusion_matrix
import recognition
import operator

model = SinglePersonSVM(weights_path="/data/liujiaji/action/kedaclassroom/kedasvmpkl/kedasvm_weights_newppx.pkl")


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
odatasets = []
actionLabels = []
front_ids = []
center_ids = []
dict = json.load(open('/data/liujiaji/action/5107_1-3-4-5-6-8-9-15.json','r',encoding='utf-8'))
for key,value in  dict.items():
    # 对应的图片
    img= Image.open('/data/liujiaji/action/kedaclassroom/5107/5107_img/'+key)
    fileprefix=re.findall(r'(.+?)\.',key)
    # 对应的骨架点
    pose = np.load('/data/liujiaji/action/kedaclassroom/5107/npy/%s.npy' %fileprefix[0] )
    pose = pose[:,:9,:2]
    testpoint = []
    ori_allpose = []
    svm_allpose = []
    for i in range(pose.shape[0]):  #（44，25，3）44个人，25个关键点，3维坐标信息x,y,confidence
        x1, y1 = pose[i][1][:2]  #脖子点
        x8, y8 = pose[i][8][:2]  #骨盆点
        xc, yc = (x1+x8)*0.5, (y1+y8)*0.5 #分辨是否在 前排 中心
        testpoint.append([xc,yc]) # 该 key 图片下的 多个人的骨架点
        ori_allpose.append(pose[i])

    # for i in range(pose.shape[0]):
    #     for j in range(len(pose[i])):
    #         pose[i][j][0] = pose[i][j][0]/1920
    #         pose[i][j][1] = pose[i][j][1]/1080

    #     svm_allpose.append(pose[i])

    # print(svm_allpose)
    # print(len(testpoint))
    ps = []
    cps = []
    ts = []
    ids = []
    for i in range(len(dict[key]['regions'])):
        id = dict[key]['regions'][i]['id']  #学生id
        ids.append(id)

        tags =  dict[key]['regions'][i]['tags']   #多个标签
        ts.append(tags)

        pointxy = []
        point = dict[key]['regions'][i]['points']  # 该 key 图片下的目标框
        for j in range(len(point)):
            pointxy.append([point[j]['x'],point[j]['y']]) # 该 key 图片下的矩形目标框
        centerp = [(pointxy[0][0]+pointxy[1][0])/2,(pointxy[0][1]+pointxy[2][1])/2]
        # print(centerp)
        cps.append(centerp)
        ps.append(pointxy) # 该 key 图片下的 多个目标框
        # print(pointxy)
    # print(ps)
    # print(ts)
    # print(len(ps))
    # print(len(ids))
    # print(len(ts))
    '''
    5107    front[[0,412],[7,704],[1920,1080],[1920,625]]
            center[[731,342],[1166,367],[751,866],[11,707]]
    '''

    for m in range(len(ps)):
        mind = 40
        index = -1
        for n in range(len(testpoint)):
            # print(testpoint[n])
            flag = isInterArea(testpoint[n],ps[m])  # 骨架点在目标框内
            flag_front = isInterArea(testpoint[n],[[0,412],[7,704],[1920,1080],[1920,625]]) #骨盆点在前排
            flag_center = isInterArea(testpoint[n],[[731,342],[1166,367],[751,866],[11,707]]) #骨盆点在前排
            if flag:
                front_ids.append(ids[m])
                # if len(ts[m]) == 1 and ts[m][0] == 'headDown':
                #     head_datasets.append(allpose[n])
                #     head_actionLabels.append(ts[m][0])
                #     datasets.append(allpose[n])
                #     actionLabels.append('unknown')
                # elif len(ts[m]) == 1 and ts[m][0] == 'headUp' :
                #     head_datasets.append(allpose[n])
                #     head_actionLabels.append(ts[m][0])
                #     datasets.append(allpose[n])
                #     actionLabels.append('listenClass')
                # else:

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
                if ts[m][o] != 'headDown' and ts[m][o] != 'headUp':
                    datasets.append(ori_allpose[index])
                    actionLabels.append(ts[m][o])
                else:
                    if ts[m][o] == 'headDown' or ts[m][o] == 'headUp'  :
                        head_datasets.append(ori_allpose[index])
                        head_actionLabels.append(ts[m][o])
                    
                      # # 8分类
                    # datasets.append(allpose[n])
                    # actionLabels.append(ts[m][o])


            # if flag == True and flag_centert == True:
            #     center_ids.append(ids[m])
    
            # print(len(front_ids),len(center_ids))

                  

    # 将前排 中心的id 写入
# print('front:',front_ids)
# print('center:',center_ids)
# np.save('/data/liujiaji/action/kedaclassroom/testKedaClass/front_id', np.array(front_ids))
# np.save('/data/liujiaji/action/kedaclassroom/testKedaClass/center_id', np.array(center_ids))
# print('finish')

    # print(key,fileprefix)


# 测试抬头低头
print(len(head_datasets))
print(len(head_actionLabels))

test_label= []
for i in range(len(head_datasets)):
    headUpDown = recognition.HeadMetric(head_datasets[i])
    if headUpDown == 1:
        test_label.append('headUp')
    else:
        test_label.append('headDown')
print(len(test_label))
print(classification_report(head_actionLabels, test_label))
print(confusion_matrix(head_actionLabels, test_label))



print(len(datasets))
# print(len(actionLabels))
# print(actionLabels)


datasets_list = []
for i in range(len(datasets)): # 共i个标签
    datasets_list.append(datasets[i].reshape(18).tolist())

    # dist10 = get_joint_distance( datasets[i][1], datasets[i][0])  #鼻子-脖子
    # dist32 = get_joint_distance( datasets[i][3], datasets[i][2]) #右肩膀-右手肘
    # dist43 = get_joint_distance( datasets[i][4], datasets[i][3]) #右手肘-右手腕
    # dist65 = get_joint_distance( datasets[i][6], datasets[i][5]) #左肩膀-左手肘
    # dist76 = get_joint_distance( datasets[i][7], datasets[i][6])  #左手肘-左手腕

    # angle10 = get_angle_north( datasets[i][1], datasets[i][0],dist10) #鼻子-脖子 相对垂直90°方向的偏移  <90°
    # angle32 = 90 + get_angle_north( datasets[i][3], datasets[i][2],dist32)#右肩膀-右手肘延申  相对水平180°方向的偏移 >90°
    # angle43 = get_angle_north( datasets[i][4], datasets[i][3],dist43)#右手肘-右手腕 相对垂直90°方向的偏移  <90°
    # angle65 = 90 + get_angle_north( datasets[i][6], datasets[i][5],dist65)#左肩膀-左手肘的延申 相对水平0°方向的偏移 >90°
    # angle76 = get_angle_north( datasets[i][7], datasets[i][6],dist76) #鼻子-脖子 相对垂直90°方向的偏移  <90°

    # action_feature = [dist10,dist32,dist43,dist65,dist76,angle10,angle32,angle43,angle65,angle76]

    # datalist = datasets[i].reshape(18).tolist()

    # for k in range(len(action_feature)):
    #     datalist.append(action_feature[k])

    # datasets_list.append(datalist)


pre_datasets_list = []
for i in range(len(datasets)):
    xx = []
    yy = []
    for j in range(datasets[i].shape[0]):  # j 9 个关键点
        # datasets[i][j][0] = datasets[i][j][0]/1920
        # datasets[i][j][1] = datasets[i][j][1]/1080
        xx.append(datasets[i][j][0])
        yy.append(datasets[i][j][1])
    # print(xx)
    # print(min(xx),max(xx))
    for j in range(datasets[i].shape[0]):
        datasets[i][j][0] = 960 * ((datasets[i][j][0] - min(xx)) / (max(xx) - min(xx)))
        datasets[i][j][1] = 540 * ((datasets[i][j][1] - min(yy)) / (max(yy) - min(yy)))
    # print(datasets[i])

    # print('//////',datasets[i][1])

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

    action_feature = [dist10,dist32,dist43,dist65,dist76,angle10,angle32,angle43,angle65,angle76]
    
    datalist = datasets[i].reshape(18).tolist()

    for k in range(len(action_feature)):
        datalist.append(action_feature[k])

    pre_datasets_list.append(datalist)

pre_test = []
for i in range(len(pre_datasets_list)):
    p_test = model.predict([pre_datasets_list[i]])
    pre_test.append(p_test)
# print(pre_test)
print(classification_report(actionLabels, pre_test))
print(confusion_matrix(actionLabels, pre_test))



# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(pre_datasets_list, actionLabels, random_state=1, train_size=0.9)
# # print(np.shape(x_train)) #(1089,75)
# # print(np.shape(x_test)) #(273,75)


# model = SinglePersonSVM()
# model.train(x_train,y_train,save_path='/data/liujiaji/action/kedaclassroom/kedasvmpkl/kedasvm_weights_8ppx.pkl')
# model.eval(x_train,y_train,report_path='/data/liujiaji/action/kedaclassroom/kedasvmpkl/kedasvm_weights_8ppx.pkl')
# model.eval(x_test,y_test,report_path='/data/liujiaji/action/kedaclassroom/kedasvmpkl/kedasvm_weights_8ppx.pkl')

# pre_test = []
# for i in range(len(x_test)):
#     p_test = model.predict([x_test[i]])
#     pre_test.append(p_test)

# print(classification_report(y_test, pre_test))
# print(confusion_matrix(y_test, pre_test))

