import sys
import numpy as np
import os
import json
import sklearn
from numpy.core.fromnumeric import reshape
sys.path.append('/data/liujiaji/action/smartcam')
from linkeaction import SinglePersonSVM
from sklearn import svm
import  cv2
from sklearn import model_selection
import  time
import  datetime
'''
### SVM测试
path_list = os.listdir('/data/liujiaji/action/kedaclassroom/5107/npy/') #   /data/liujiaji/action/keda/npy/
path_list.sort(key = lambda x : x[:-4])
for file in path_list:
    # print(file)
    # pose = np.load('/data/liujiaji/action/skeleton_16/results/c1_front_9910.npy') #训练集
    pose = np.load('/data/liujiaji/action/kedaclassroom/5107/npy/'+file) #训练集
    pose = pose[:,:8,:]
    # print(pose.shape)

    ####### 改变输入，相对坐标
    # new_pose = []
    # for o in range(pose.shape[0]): #多少个骨架 10
    #     temp = []
    #     sum_x,sum_y = 0,0
    #     for s in range(pose.shape[1]): #一个骨架 25个关键点
    #         temp.append(pose[o][s])
    #         sum_x += temp[s][0]
    #         sum_y += temp[s][1]
    #         avg_x = sum_x/len(temp)
    #         avg_y = sum_y/len(temp)
    #         if pose[o][s].any() != 0:
    #             nep = pose[o][s] - pose[o][8]  #其余点相对骨盆的新的坐标信息
    #             new_pose.insert(s,nep) #所有人检测到的骨架的 相对x,y坐标
    #         else:
    #             nep = [avg_x,avg_y] - pose[o][8]
    #             new_pose.insert(s,nep)
    # newpose = np.array(new_pose)
    # newpose.resize(pose.shape[0],25,2)
    #######
    # newpose_array.resize(pose.shape[0],50)

    pp = []
    # model = SinglePersonSVM(weights_path="/data/workspace/yaojiapeng/dangxiao/smartcam/weights/pose_svm_dangxiao_demo.pkl")
    model = SinglePersonSVM(weights_path="/data/liujiaji/action/svm_weights.pkl")
    for i in range(0,pose.shape[0]):
        # print(pose[i])
        # print(type(pose[i]))
        # print(pose[i].shape)
        idx2act = ["", "sleep", "raise hand", "take note", "use phone"]
        # pose: 25x2
        #用相对坐标
        # label = idx2act[model.predict(newpose[i].reshape(50))]

        #原始坐标
        label = idx2act[model.predict(pose[i].reshape(24))]
        # pp.append([i,label])
        pp.append([pose[i][0][:2],label])
        # dict = {label,pose[i][0][:2]}
    # print(pp) # 每个npy文件的 action标签
    # print(len(pp))

    img_src = cv2.imread('/data/liujiaji/action/kedaclassroom/5107/5107_img/%s.jpg' % (os.path.splitext(file)[0]))
    for i in range(len(pp)):
        # print(tuple(pp[i][0]))
        if pp[i][1] == '':
            cv2.putText(img_src,pp[i][1],tuple(pp[i][0]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        if pp[i][1] == 'sleep':
            cv2.putText(img_src,pp[i][1],tuple(pp[i][0]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if pp[i][1] == 'raise hand':
            cv2.putText(img_src,pp[i][1],tuple(pp[i][0]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        if pp[i][1] == 'take note':
            cv2.putText(img_src,pp[i][1],tuple(pp[i][0]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        if pp[i][1] == 'use phone':
            cv2.putText(img_src,pp[i][1],tuple(pp[i][0]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imwrite('/data/liujiaji/action/kedaclassroom/5107/visAction/%s.jpg' % (os.path.splitext(file)[0]),img_src)


    print_log=open("/data/liujiaji/action/5107action.txt",'a+')
    # print(file, pp, file=print_log)
    print(file, pp, file=print_log)

    
# data = []
# for line in open("action.txt","r"): #设置文件对象并读取每一行文件
#     data.append(line)               #将每一行文件加入到list中
# print(data[0])
# print(type(data[0]))
# print(data[0][0])

'''

### SVM训练
path_list = os.listdir('/data/liujiaji/action/action_train/')
all_pose = []
all_sebe_label = []
posefilename = []
actionfilename = []
for file in path_list:
    if os.path.splitext(file)[1] == '.npy':
        listname = os.path.splitext(file)[0]
        posefilename.append(listname)
    if os.path.splitext(file)[1] == '.json':
        listname = os.path.splitext(file)[0]
        actionfilename.append(listname)
if(len(posefilename) == len(actionfilename)):
    for m in range(len(posefilename)):
        for n in range(len(actionfilename)):
            if posefilename[m] == actionfilename[n]:
                # print('same filename:',posefilename[m],actionfilename[n])
                # npy文件 [id,25,3]
                pose_load = '/data/liujiaji/action/action_train/'+posefilename[m]+'.npy'
                pose = np.load(pose_load) #(N,25,3)
                # pose = pose[:,:,:2]  ##(N,25,2)
                
                # 预处理：上半身关键点 [0-7]
                # pose = pose[:,:8,:]

                # print('1.',pose.shape)

                # 取上半身节点后，归一化：[x/image_width],y/image_height]  1440*2560
                # for i in range(pose.shape[0]): #共有几个人
                #     xx = pose[i][:,0]
                #     yy = pose[i][:,1]
                    
                #     poselist = pose[i].tolist()
                #     # print(poselist)
                #     normal_poselist = []
                #     for j in range(8):
                #         normal_x = (poselist[j][0] - min(xx))/(max(xx) - min(xx))
                #         normal_y = (poselist[j][1] - min(yy))/(max(yy) - min(yy))
                #         normal_poselist.append([normal_x,normal_y,poselist[j][2]])
                # print(normal_poselist)
                # normal_pose = np.array(normal_poselist)
                # # print('2.',normal_pose.shape)
                # normal_pose.resize(pose.shape[0],24)
                # # print('3.',normal_pose.shape)
                # all_pose.append(normal_pose.tolist())



                ######## 骨架的相对坐标
                # new_pose  = []
                # for o in range(pose.shape[0]): #多少个骨架  10个
                #     temp = []
                #     sum_x,sum_y = 0,0
                #     for s in range(pose.shape[1]): #25个骨骼点
                #         if pose[o][s].any() != 0:
                #             temp.append(pose[o][s])
                #             for t in range(len(temp)):
                #                 sum_x += temp[t][0]
                #                 sum_y += temp[t][1]
                #                 avg_x = sum_x/len(temp)
                #                 avg_y = sum_y/len(temp)
                #             nep = pose[o][s] - pose[o][8]  #其余点相对骨盆的新的坐标信息
                #             # print('第',o,'个人的所有骨架的相对点:',nep)
                #             new_pose.insert(s,nep) #所有人检测到的骨架的 相对x,y坐标
                #         else:
                #             nep = [avg_x,avg_y] - pose[o][8]
                #             new_pose.insert(s,nep)
                # newpose = np.array(new_pose)   # (NX25,2)
                # newpose.resize(pose.shape[0],50) #(N,25,2)
                # all_pose.append(newpose.tolist())
                #######
                
                # 原坐标，8个关键点
                repose = np.reshape(pose,(-1,75)) #(N, 25 or 8 * 2 or 3)  array
                all_pose.append(repose.tolist())


                # [骨骼id,action]
                sebe = json.load(open('/data/liujiaji/action/action_train/'+actionfilename[n]+'.json','r'))
                sort_sebe = dict(sorted(sebe.items(),key=lambda e:e[0]))
                # print(sort_sebe)
                list_sebe_values = [l for l in sort_sebe.values()]
                label = []
                for k in range(len(list_sebe_values)):
                    if list_sebe_values[k] == '其他' or list_sebe_values[k] == '':
                        label.append(0)
                    elif list_sebe_values[k] == '睡觉':
                        label.append(1)
                    elif list_sebe_values[k] == '举手':
                        label.append(2)
                    elif list_sebe_values[k] == '记笔记':
                        label.append(3)
                    else:
                        list_sebe_values[k] == '玩手机'
                        label.append(4)
                for p in range(pose.shape[0]):
                    if str(p) not in sort_sebe.keys():
                        list_sebe_values.insert(p,'')
                        label.insert(p,0)
                all_sebe_label.append(label)  
                break
# print(all_pose)
pose_list = []
for i in range(len(all_pose)):
    for j in range(len(all_pose[i])):
        pose_list.append(all_pose[i][j])
# print(pose_list)
print('样本数pose:',len(pose_list))
label_list = []
for m in range(len(all_sebe_label)):
    for n in range(len(all_sebe_label[m])):
        label_list.append(all_sebe_label[m][n])
# print(label_list)    
print('样本数label:',len(label_list))

#### 训练集的标签占比
other,sleep,raisehand,takenote,usephone = 0,0,0,0,0
for i in range(len(label_list)):
    if label_list[i] == 0:
        other += 1
    elif label_list[i] == 1:
        sleep += 1
    elif label_list[i] == 2:
        raisehand += 1
    elif label_list[i] == 3:
        takenote += 1
    else:
        label_list[i] == 4
        usephone += 1
print('other:',other,'sleep:',sleep,'raisehand:',raisehand,'takenote:',takenote,'usephone:',usephone)
#####

####### 标签均衡：400个
## len(pose_list) = len(label_list) 一一对应 : 第n个骨架---> 第n个动作
temp_label = []
temp_pose = []
otherC,sleepC,raisehandC,takenoteC,usephoneC = 0,0,0,0,0
for n in range(len(label_list)):
    # temp_label = []
    # temp_pose = []
    if label_list[n] == 0 :
        if otherC < 659:
            temp_label.append(label_list[n])
            temp_pose.append(pose_list[n])
            otherC += 1
    if label_list[n] == 1:
        if sleepC < 659:
            temp_label.append(label_list[n])
            temp_pose.append(pose_list[n])
            sleepC += 1
    if label_list[n] == 2:
        if raisehandC < 659:
            temp_label.append(label_list[n])
            temp_pose.append(pose_list[n])
            raisehandC += 1
    if label_list[n] == 3:
        if takenoteC < 659:
            temp_label.append(label_list[n])
            temp_pose.append(pose_list[n])
            takenoteC += 1
    if label_list[n] == 4:
        if usephoneC < 659:
            temp_label.append(label_list[n])
            temp_pose.append(pose_list[n])
            usephoneC += 1
    # temp_label_list.append(temp_label)

print('均衡后pose:',len(temp_pose))
print('均衡后label:',len(temp_label))


# 划分测试和训练集 比例为0.7
# 数据不均衡
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(pose_list, label_list, random_state=1, train_size=0.7)

# 数据均衡
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(temp_pose, temp_label, random_state=1, train_size=0.6)

# Kernel = ["linear","poly","rbf","sigmoid"]
# for kernel in Kernel:
#     clf= svm.SVC(kernel = kernel, gamma="auto", degree = 1, cache_size=5000).fit(x_train,y_train)
#     pickle.dump(clf,open('/data/liujiaji/action/svm1_weights.pkl','wb'))
#     print("Weights saved to  /data/liujiaji/action/svm1_weights.pkl ")
#     print("The accuracy under kernel %s is %f" % (kernel,clf.score(x_test,y_test)))

model = SinglePersonSVM()
model.train(x_train,y_train,save_path='/data/liujiaji/action/svm_weights.pkl')
model.eval(x_train,y_train,report_path='/data/liujiaji/action/svm_weights.pkl')
model.eval(x_test,y_test,report_path='/data/liujiaji/action/svm_weights.pkl')




###### 查看权重文件
# import pickle
# path = '/data/liujiaji/action/svm_weights.pkl'  #pkl文件所在路径
# # path = '/data/workspace/yaojiapeng/dangxiao/smartcam/weights/pose_svm_dangxiao_demo.pkl'   
# f=open(path,'rb')
# data=pickle.load(f)
# print(data)
