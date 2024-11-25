
import skeleton
import os
import cv2
from PIL import Image
import numpy as np



def carve_box(one_skeleton, height, width):
    min_x = min_y = float('inf')  #正无穷
    max_x = max_y = 0 #0
    
    for point in one_skeleton: #25个点 [x,y]
        if point[0] == 0 or point[1] == 0:
            continue
        if point[1] < min_y:
            min_y = point[1]
        if point[0] > max_x:
            max_x = point[0]
        if point[1] > max_y:
            max_y = point[1]
        if point[0] < min_x:
            min_x = point[0]
    if one_skeleton[1][1]*one_skeleton[8][1] != 0:
        H_Exp = int((one_skeleton[8][1] - one_skeleton[1][1])/2)
    else:
        H_Exp = 0
    if one_skeleton[1][0]*one_skeleton[2][0] != 0:
        W_Exp = one_skeleton[1][0] - one_skeleton[2][0]
    else:
        W_Exp = 0
    min_y = min_y - H_Exp
    max_y = max_y + H_Exp
    min_x = min_x - W_Exp
    max_x = max_x + W_Exp

    if one_skeleton[8][1] == 0:
        # continue
        pass
    # max_y = one_skeleton[8][1]
    if min_x < 0:
        min_x = 0
    if max_x > width:
        max_x = width
    if min_y < 0:
        min_y = 0
    if max_y > height:
        max_y = height
    # return min_x, max_x, min_y, max_y

    return int(min_x), int(max_x), int(min_y), int(max_y)


def process_img(img_path,cropimg_path):
    img_list = []
    vis_img_point = []
    for kp in posekeypoints:
        img = Image.open(img_path)
        min_x, max_x, min_y, max_y = carve_box(kp, img.size[1], img.size[0])
        if max_x - min_x > 40 and max_y - min_y > 40 :
            print(i,min_x, max_x, min_y, max_y)
            cropped_img = img.crop((min_x,min_y,max_x,max_y))
            cropped_img.save(cropimg_path)

            vis_xx = (min_x + max_x)//2
            vis_yy = (min_y + max_y)//2
            vis_img_point.append([vis_xx,vis_yy])
            img_list.append(cropped_img)
    return img_list,vis_img_point

os.environ["CUDA_VISIBLE_DEVICES"]="2"
opper = skeleton.OpenPose()
pathlist = os.listdir('/data/liujiaji/action/kedaclassroom/5107/crop_img/unknown/')
i = 0
for file in pathlist:
    img = cv2.imread('/data/liujiaji/action/kedaclassroom/5107/crop_img/unknown/'+file)
    datum = opper.infer(img)
    posekeypoints = datum.poseKeypoints
    if posekeypoints is None:
        continue
    i = i+1
    if i %3  != 0:
        process_img('/data/liujiaji/action/kedaclassroom/5107/crop_img/unknown/'+file,'/data/liujiaji/action/kedaclassroom/vggimg/train/unknown/'+file)
    else:
        process_img('/data/liujiaji/action/kedaclassroom/5107/crop_img/unknown/'+file,'/data/liujiaji/action/kedaclassroom/vggimg/test/unknown/'+file)
print('********',i)
print('finish')