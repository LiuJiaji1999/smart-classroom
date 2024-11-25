
import numpy as np
import math
import cv2
import skeleton
import os


# posekeypoints = np.load('/data/liujiaji/action/skeleton_action/npy/keda_20223241103504484.npy')

# 预处理：上半身关键点 [0-7]
def preprocess_pose(posekeypoints):

    pose8_array = posekeypoints[:,:8,:]
    print(pose8_array.shape)
    # print(pose8_array)
    pose8 = np.reshape(pose8_array,(-1,3))
    pose = pose8.tolist()
    # print(pose)
    return pose

# 取上半身节点，并归一化：[x/image_width],y/image_height]
def normalized_pose(img,posekeypoints):
    pose = preprocess_pose(posekeypoints)
    # img = cv2.imread('/data/liujiaji/action/skeleton_action/img/keda_20223241103504484.jpg')
    print(img.shape[1],img.shape[0])
    normal_pose = []
    for l in range(8):
        # normal_pose.append([pose[l][0]/img.shape[1],pose[l][1]/img.shape[0],pose[l][2]])
        normal_pose.append([pose[l][0]*img.shape[1],pose[l][1]*img.shape[0],pose[l][2]])

    # # 用于可视化
    # cv2.circle(img, tuple([int(pose8[0][0]),int(pose8[0][1])]),2,(0,255,0), 3)
    # cv2.line(img, tuple(pose8[1]), tuple(pose8[0]), (0, 255, 0), 3)
    # cv2.line(img, tuple(pose8[3]), tuple(pose8[2]), (0, 0, 255), 3)
    # cv2.line(img, tuple(pose8[4]), tuple(pose8[3]), (255, 3, 0), 3)
    # cv2.line(img, tuple(pose8[5]), tuple(pose8[6]), (255, 100, 0), 3)
    # cv2.line(img, tuple(pose8[6]), tuple(pose8[7]), (0, 255, 255), 3)
    # cv2.imwrite('/data/liujiaji/action/skeleton_action/img/vis.jpg',img)
    
    # print(normal_pose)
    return normal_pose


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


# 26维特征向量
def make_datasets(posekeypoints):
    action_represent = []

    normal_pose = normalized_pose(img,posekeypoints) # 0-7关键点的[x,y]
    dist10 = get_joint_distance(normal_pose[1],normal_pose[0])  #鼻子-脖子
    dist32 = get_joint_distance(normal_pose[3],normal_pose[2]) #右肩膀-右手肘
    dist43 = get_joint_distance(normal_pose[4],normal_pose[3]) #右手肘-右手腕
    dist65 = get_joint_distance(normal_pose[6],normal_pose[5]) #左肩膀-左手肘
    dist76 = get_joint_distance(normal_pose[7],normal_pose[6])  #左手肘-左手腕

    angle10 = get_angle_north(normal_pose[1],normal_pose[0],dist10) #鼻子-脖子 相对垂直90°方向的偏移  <90°
    angle32 = 90 + get_angle_north(normal_pose[3],normal_pose[2],dist32)#右肩膀-右手肘延申  相对水平180°方向的偏移 >90°
    angle43 = get_angle_north(normal_pose[4],normal_pose[3],dist43)#右手肘-右手腕 相对垂直90°方向的偏移  <90°
    angle65 = 90 + get_angle_north(normal_pose[6],normal_pose[5],dist65)#左肩膀-左手肘的延申 相对水平0°方向的偏移 >90°
    angle76 = get_angle_north(normal_pose[7],normal_pose[6],dist76) #鼻子-脖子 相对垂直90°方向的偏移  <90°

    # action_feature = [dist10,dist32,dist43,dist65,dist76,angle10,angle32,angle43,angle65,angle76]
    action_feature = [angle10,angle32,angle43,angle65,angle76]
    for i in range(8):
        for j in range(3):
            action_represent.append(normal_pose[i][j])
    # for k in range(len(action_feature)):
    #     action_represent.append(action_feature[k])
    # print(len(action_represent),'维特征向量:',action_represent)
    return action_represent


def make_datasets_nothing(posekeypoints):
    new_pose_list = []
    pose_list = preprocess_pose(posekeypoints)
    for i in range(8):
        for j in range(3):
            new_pose_list.append(pose_list[i][j])
    # print(len(new_pose_list),'维特征向量:',new_pose_list)
    return new_pose_list







os.environ["CUDA_VISIBLE_DEVICES"]="2"
opper = skeleton.OpenPose()
action_represents = []
action_labels = []

pathlist = os.listdir('/data/liujiaji/action/kedaclassroom/5107/crop_img/headDown/')
for file in pathlist:

    img = cv2.imread('/data/liujiaji/action/kedaclassroom/5107/crop_img/headDown/'+file)
    datum = opper.infer(img)
    posekeypoints = datum.poseKeypoints
    if posekeypoints is None:
        continue

    # action_represent = make_datasets(posekeypoints)
    action_represent = make_datasets_nothing(posekeypoints)

    action_represents.append(action_represent)
# print(len(action_represents),'维特征向量:',action_represents)
for i in range(len(action_represents)):
    action_labels.append('headDown')
    with open("/data/liujiaji/action/kedaclassroom/highallpoiang_actionDatasets.txt","a+") as file:
        file.write(str(action_represents[i]))
        file.write('\n')# 换行
    with open("/data/liujiaji/action/kedaclassroom/highallpoiang_actionDatasetsLabel.txt","a+") as file:
        file.write(action_labels[i])
        file.write('\n')# 换行
print('finish')




